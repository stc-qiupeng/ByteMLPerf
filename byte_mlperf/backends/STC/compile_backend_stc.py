# Copyright 2023 Stream Computing Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
compile backend of stc
"""

from distutils.command.config import config
import os
import ast
import sys
import json
import shutil
import atexit
import logging
import subprocess

from tvm import relay
import tensorflow as tf
from pathlib import Path
from typing import Any, Dict
import tvm
import onnx
import numpy as np
import onnxruntime as ort
from onnx import shape_inference

import tvm.relay.testing.tf as tf_testing
from tb.relay.backend import build_nosch_mod
from tb.relay.ilp_graph_schedule.sch_batch import infer_opt_batches
from stc_ddk.stc_aic.frontends.onnx import load_onnx
from stc_ddk.stc_aic.stc_aic import pt2onnx
from tb.relay.graph_schedule import GetNPUSubFunctions
from byte_mlperf.backends import compile_backend
from .runtime_backend_stc import INPUT_TYPE
from tensorflow.core.framework import tensor_shape_pb2


sys.path.append(os.path.dirname(__file__))
log = logging.getLogger("CompileBackendSTC")
log.setLevel(logging.INFO)

seq = tvm.transform.Sequential(
    [
        tvm.relay.transform.RemoveUnusedFunctions(),
        tvm.relay.transform.ConvertLayout(
            {
                "nn.conv2d": ["NHWC", "HWIO"],
                "nn.global_avg_pool2d": ["NHWC", "HWIO"],
                "nn.max_pool2d": ["NHWC", "HWIO"],
                "nn.conv2d_transpose": ["NHWC", "HWIO"],
                "nn.avg_pool2d": ["NHWC", "HWIO"],
                "nn.adaptive_avg_pool1d": ["NWC"],
                "image.resize2d": ["NHWC"],
            }
        ),
    ]
)


class CompileBackendSTC(compile_backend.CompileBackend):
    """
    STC compile backend.
    """

    TARGET = "stc_tc"

    def __init__(self):
        super().__init__()
        self.frontend_util = None
        self.need_quant = False
        self.hardware_type = "STC"
        self.stc_dtype = "float16"
        self.object_suffix = ".stcobj"
        self.best_batch = 1
        self.batch_sizes = []
        self.tmpdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "mix_tmp")
        self.tmpfiles = set()
        atexit.register(self.__del)

    def __del(self):
        if self.frontend_util is not None:
            self.tmpfiles.update(self.frontend_util.gc())
        for tmpfile in self.tmpfiles:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)

    def get_interact_profile(self, config):
        return []

    def version(self):
        return "2.3"

    def parse_tf(self, model_format, model, input_shape_dict, input_dtype_dict, output_names):
        """Parse tensorflow model."""
        graph_def = tf.compat.v1.GraphDef()
        if model_format == "pb":
            with tf.compat.v1.gfile.GFile(model, "rb") as f:
                graph_def.ParseFromString(f.read())
        elif model_format == "saved_model":
            dtype = "float32"
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, ["serve"], model)
                    graph_def = sess.graph_def

        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        input_map = {}
        tf.compat.v1.disable_eager_execution()
        for input_name, input_shape in input_shape_dict.items():
            input_map[input_name] = tf.compat.v1.placeholder(
                shape=input_shape,
                dtype=INPUT_TYPE[input_dtype_dict[input_name]],
                name=input_name.split(":")[0],
            )
        tf.import_graph_def(graph_def, name="", input_map=input_map)
        with tf.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(
                sess, output_names.split(",")[0].split(":")[0]
            )
        relay_mod, params = relay.frontend.from_tensorflow(
            graph_def, outputs=output_names.split(",")
        )
        return relay_mod, params

    def parse_onnx(self, onnx_model_path, input_dict, input_dtype_dict):
        """Parse onnx model."""
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = shape_inference.infer_shapes(onnx_model)

        def is_prime_number(number: int):
            if number <= 1:
                return False
            i = 2
            while i * i <= number:
                if number % i == 0:
                    return False
                i += 1
            return True

        def find_unique_batch(model_shapes_set, batch_label=65535):
            # find unique batch for inferring best batch
            while 0 in model_shapes_set:
                model_shapes_set.remove(0)
            for i in range(2, batch_label):
                unique_flag = True
                if is_prime_number(i): 
                    for dim in model_shapes_set:
                        if i == dim or dim % i == 0:
                            unique_flag = False
                else:
                    unique_flag = False
                if unique_flag:
                    return i
            raise ValueError("Cannot find an appropriate value for batch_label!")

        for node in onnx_model.graph.node:
            for output in node.output:
                onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        ort_session = ort.InferenceSession(onnx_model.SerializeToString())
        feeds_dict = {}
        model_shapes = []
        for input_name, input_shape in input_dict.items():
            feeds_dict[input_name] = np.array(
                np.random.random(input_shape), dtype=INPUT_TYPE[input_dtype_dict[input_name]]
            )
            model_shapes.extend(list(input_shape))
        
        outputs = [x.name for x in ort_session.get_outputs()]
        ort_outs = ort_session.run(outputs, feeds_dict) 
        for ort_out in ort_outs:
            model_shapes.extend(list(ort_out.shape))
        
        unique_batch = find_unique_batch(set(model_shapes))
        log.info("onnx find unique batch:{}".format(unique_batch))
        unique_batch_input_dict = {}
        for input_name, input_shape in input_dict.items():
            unique_batch_input_dict[input_name] = [
                unique_batch if list(input_shape)[0] == 1 else list(input_shape)[0]
            ] + list(input_shape)[1:]
        print(unique_batch_input_dict)
        
        relay_mod, params = tvm.relay.frontend.from_onnx(
            onnx.load(onnx_model_path), shape=unique_batch_input_dict
        )
        return relay_mod, params, unique_batch

    def infer_best_batch(self, relay_mod, params, batch_label: int = 65535) -> int:
        """infer best batch.

        Args:
            relay_mod: A relay model by from_frontend
            params: params by from_frontend
            input_dict: {input_name: input_shape},  input_name:str   input_shape:tuple/list

        Returns
        ret: The best batch.
        """
        with tvm.transform.PassContext(opt_level=3):
            relay_mod = seq(relay_mod)
        
        build_configure = {
            "tb.ncore": 8,
        }
        ps_ctx = tvm.transform.PassContext(config=build_configure, opt_level=3)
        mod = build_nosch_mod(relay_mod, params, "stc_tc", ps_ctx)
        func = GetNPUSubFunctions(mod["main"])[0]
        best_batch = infer_opt_batches(func, batch_label=batch_label)
        log.info("infered best_batch: %d", best_batch)
        return best_batch

    def pre_optimize(self, configs: Dict[str, Any]):
        logging.root.level = logging.WARNING
        log.info("Running Backend Pre Compilation...")
        self.model_name = configs["model_info"]["model"]
        self.model_path = configs["model_info"]["model_path"]
        framework = configs["model_info"]["framework"].lower()
        model_format = configs["model_info"]["model_format"]
        self.input_names = configs["model_info"]["inputs"]
        self.input_dtypes = configs["model_info"]["input_type"]

        d_args = {}
        d_args["filename"] = self.model_path
        d_args["input_name_shapes"] = configs["model_info"]["input_shape"]
        d_args["output_names"] = configs["model_info"]["outputs"]
        d_args["input_name_dtypes"] = {}
        for item in zip(self.input_names.split(","), self.input_dtypes.split(",")):
            d_args["input_name_dtypes"][item[0]] = item[1]

        to_onnx = False
        if framework == "tensorflow":
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    meta_graph_def = tf.compat.v1.saved_model.loader.load(
                        sess, ["serve"], self.model_path
                    )
                    signature_def = meta_graph_def.signature_def["serving_default"]
                    for _, _input in signature_def.inputs.items():
                        if "StatefulPartitionedCall" in _input.name:
                            to_onnx = True
                            break
                    if not to_onnx:
                        for _, output in signature_def.outputs.items():
                            if "StatefulPartitionedCall" in output.name:
                                to_onnx = True
                                break
            if to_onnx is True:
                self.model_path = d_args["filename"] + ".onnx"
                if os.path.exists(self.model_path):
                    log.info("The saved_model has already converted to onnx.")
                else:
                    load_onnx(d_args)
        elif framework == "pytorch":
            to_onnx = True
            self.model_path = d_args["filename"].split(".")[0] + ".onnx"
            if os.path.exists(self.model_path):
                log.info("The pytorch model has already converted to onnx.")
            else:
                pt2onnx(d_args)
        if to_onnx is True:
            framework = "onnx"

        best_batch = 1
        tf_input_shape_dict = {}
        for input_name, input_shape in d_args["input_name_shapes"].items():
            tf_input_shape_dict[input_name] = [65535 * input_shape[0]] + input_shape[1:]

        log.info("Parsing model and infering the best batch...")
        if framework == "tensorflow":
            relay_mod, params = self.parse_tf(
                model_format,
                self.model_path,
                tf_input_shape_dict,
                d_args["input_name_dtypes"],
                d_args["output_names"],
            )
            best_batch = self.infer_best_batch(relay_mod, params)
        elif framework == "onnx":
            relay_mod, params, unique_batch = self.parse_onnx(
                self.model_path, d_args["input_name_shapes"], d_args["input_name_dtypes"]
            )
            best_batch = self.infer_best_batch(relay_mod, params, batch_label=unique_batch)

        log.info("best_batch: %d", best_batch)
        self.best_batch = best_batch

        for scale in [1/4, 1/2, 1, 2, 4]:
            if int(best_batch * scale) != 0:
                self.batch_sizes.append(int(best_batch * scale))
        # self.batch_sizes.append(self.best_batch)
        configs["infer_batch_sizes"] = self.batch_sizes
        print(configs["infer_batch_sizes"])
        return configs

    def get_best_batch_size(self):
        if self.best_batch is None:
            log.error("Not found the best batch_size. Please call pre_optimize to infer it.")
        return self.best_batch

    def __split_expression(self, expression):
        table = {"(", ")", "+", "-", "*", "/", " "}

        def check(words):
            for word in words:
                if word in table or word.isdigit():
                    return False
            return True

        def recursion(words):
            if not words:
                return []
            if check(words):
                return [
                    words,
                ]

            for i, word in enumerate(words):
                if word in table:
                    return recursion(words[:i]) + recursion(words[i + 1 :])
            return []

        return list(set(recursion(expression)))

    def compile(self, configs: Dict[str, Any], dataloader=None) -> Dict[str, Any]:
        logging.root.level = logging.WARNING
        log.info("Running Backend Compilation...")
        model_name = configs["model_info"]["model"]

        def gen_mix_cmd(bs):
            input_shapes = []
            outputs = ""

            for input_name in self.input_names.split(","):
                shapes = configs["model_info"]["input_shape"][input_name]
                new_shape = []
                for shape in shapes:
                    if isinstance(type, str):
                        for name in self.__split_expression(shape):
                            if name not in configs["model_info"]:
                                log.error("Not found %s in configs", name)
                            shape = shape.replace(name, f'config["model_info"]["{name}"]')
                        shape = ast.literal_eval(shape)
                    new_shape.append(shape)
                new_shape[0] *= bs
                input_shapes.append("[" + ",".join(str(val) for val in new_shape) + "]")

            input_shapes = ",".join(val for val in input_shapes)
            outputs = configs["model_info"]["outputs"]
            output_num = len(configs["model_info"]["outputs"].split(","))
            output_dtypes = ",".join(
                configs["model_info"]["model_precision"] for _ in range(output_num)
            )

            res_path = os.path.join(self.tmpdir, model_name, "bs_{}".format(bs))

            out_cmd = [
                "stc_ddk.stc_aic",
                "--model",
                self.model_path,
                "--input_names",
                self.input_names,
                "--input_shapes",
                input_shapes,
                "--input_dtypes",
                self.input_dtypes,
                "--output_names",
                outputs,
                "--output_dtypes",
                output_dtypes,
                "--outdir",
                res_path,
            ]

            return out_cmd, res_path

        for bs in configs["infer_batch_sizes"]:
            log.info("Compiling the model with batch {}".format(bs))
            out_cmd, res_path = gen_mix_cmd(bs)

            if os.path.exists(os.path.join(res_path, "model.json")):
                log.info("BS_{} stcobj has exists, skip compile.".format(bs))
            else:
                if os.path.exists(res_path):
                    shutil.rmtree(res_path)
                try:
                    log.info(" ".join(str(val) for val in out_cmd))
                    subprocess.call(out_cmd)
                except Exception:
                    pass
        
        res_path = os.path.join(self.tmpdir, model_name, "bs_{}".format(self.best_batch))
        with open(os.path.join(res_path, "model.json")) as file_reader:
            compiled_model_info = json.loads(file_reader.read())
        compile_info = {
            "model": configs["model_info"]["model"],
            "framework": configs["model_info"]["framework"],
            "compile_precision": "fp16",
            "input_type": configs["model_info"]["input_type"],
            "max_batch_size": configs["infer_batch_sizes"][-1],
            "sg_percent": compiled_model_info["stcop_rate"],
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": configs["model_info"]["input_shape"],
                    "output_tensor_map": configs["model_info"]["outputs"],
                    "compiled_model": [
                        {
                            "compiled_bs": self.best_batch,
                            "compiled_obj": configs["model_info"]["model_path"],
                        },
                    ],
                },
            ],
        }
        self.workload = configs["workload"]
        self.model_info = configs["model_info"]

        if not os.path.exists(res_path) or self.__check_aic(res_path):
            run_cmd = " ".join(str(val) for val in out_cmd)
            log.error("model convert error. run_cmd is : %s", run_cmd)
            compile_info["compile_status"] = "failed"
        else:
            compile_info["compile_status"] = "success"
        return compile_info

    def __check_aic(self, res_path):
        """Check whether the compilation was successful."""
        aic_fail_flag = False
        res_path = Path(res_path)
        json_file = res_path / "model.json"
        if json_file.exists():
            with open(str(json_file), "r") as file_reader:
                model_info = json.load(file_reader)
            if len(model_info["nodes"]) > 0:
                for node in model_info["nodes"]:
                    file = res_path / node["source"]
                    if not file.exists():
                        aic_fail_flag = True
                        break
            else:
                aic_fail_flag = True
        else:
            aic_fail_flag = True
        return aic_fail_flag
