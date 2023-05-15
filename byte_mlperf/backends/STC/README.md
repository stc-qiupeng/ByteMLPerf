<div align="center">
  <img src="STC.jpg">
</div>


# Supported model inference results
| Model name | QPS | Dataset | Metric name | Metric value |
| :-----:| :----: | :----: | :----: | :----: |
| albert-torch-fp32 | 824.49 | Open Squad 1.1 | F1 Score | 87.66 |
| bert-tf-fp32 | 822.38 | Open Squad 1.1 | F1 Score | 86.45 |
| bert-torch-fp32 | 813.86 | Open Squad 1.1 | F1 Score | 86.14 |
| resnet50-tf-fp32 | 8725.94 | Open ImageNet | Top-1 | 77.24% |
| robert-torch-fp32 | 800.7 | Open Squad 1.1 | F1 Score | 83.19 |
| widedeep-tf-fp32 | 2395899.9 | Open Criteo Kaggle | Top-1 | 77.39% |


For more detailed result information, see byte_mlperf/reports/STC/. Above model inference based on the chip named "STC P920" and the following software.

| Software | Version | Description |
| :-----:| :----: | :----: |
| HPE | 1.5.1 | Heterogeneous Programming Environment |
| TB | 1.11.0 | TensorTurbo, AI compiler developed based on TVM |
| STC_DDk | 1.1.0 | Model compilation and deployment tools developed based on TensorTurbo |


In addition to the above software, we have developed some very useful NPU(Neural-network Processing Unit) tools for monitoring the status of hardware and software, debugging bugs, analyzing accuracy and performance, as follows.

| Software  | Description |
| :-----:| :----: |
| stc-smi | Stream Computing System Management Interface for managing and monitoring NPU devices, including viewing device information and resource usage |
| stc-gdb | Stream Computing Debugger for debugging heterogeneous NPU programs  |
| stc-prof | Stream Computing Profiler, for performance analysis and optimization of heterogeneous programs  |
| stc-hpaa | Stream Computing Half-Precision Accuracy Analysis, for locating the calculation error location and corresponding data  |
| ScheduleViewer | For parsing Relay IR, exporting in JSON format so that it can be opened with Netron to view graph structure  |


See the link for more detailed software information: https://docs.streamcomputing.com/zh/latest/


# Company introduction
Beijing Stream Computing Technology Co., LTD, we are committed to providing cloud service manufacturers with high cost performance and high versatility of AI accelerated chips.

The first-generation chip achieves 128 TFLOPS in semi-precision floating-point operations, twice as big as T4. At present, the first-generation has been mass production capacity, and has completed small batch shipments to users. The second-generation chip is coming soon. 

# What we can do
We can use the AI compiler(TensorTurbo) to convert the deep learning model into an object file that can be executed on an NPU, with many accelerated optimizations involved in the conversion process, then feed input data, execute the object file on the NPU device, get the model result.

We have supported over 150 open source models(application fields include CV, NLP, recommendation, speech, OCR, multimodel), over 160 operations, and four deep learning frameworks including tensorflow 1.x and 2.x, pytorch, onnx, paddlepaddle. Most of models can achieve 2x T4 performance.


# Contact us
If you are interested in further information about the product, please contact the email: johnson@streamcomputing.com

