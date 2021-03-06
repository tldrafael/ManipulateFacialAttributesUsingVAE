?	?????|n@?????|n@!?????|n@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?????|n@DQ?O??$@1??J
?k@Ah?,{??I=??%@rEagerKernelExecute 0*	?O??6??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?/h!#/@!/] ?fS@)?/h!#/@1/] ?fS@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?'G??@!Ś????5@)?'G??@1Ś????5@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??&??@!O??96@)H5???:??1m??x??:Preprocessing2F
Iterator::Model{?"0??@!T??:jP6@)???I??1R?Ρɶ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap^K?=+/@!?Iq?kS@)?7??w??1?WO?Å??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???QC!@Q>gÕ?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	DQ?O??$@DQ?O??$@!DQ?O??$@      ??!       "	??J
?k@??J
?k@!??J
?k@*      ??!       2	h?,{??h?,{??!h?,{??:	=??%@=??%@!=??%@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???QC!@y>gÕ?V@?"S
*model/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput?Q??'??!?Q??'??"R
)model/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput?#{????!L????"e
9gradient_tape/model/conv2d_28/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterA?0?j??!\???)???0"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?}?Z]??!ՏV=????0"x
Lgradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter߲?(
E??!F޼ca???0"y
Mgradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterd?~?7@??!?,ch???0"]
?gradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DConv2D##?zk=??!D??????0"\
>gradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DConv2Ds??s_8??!??????0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterꅅK_???!:R?ˣ??0"e
9gradient_tape/model/conv2d_29/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???[???!t???n??0Q      Y@YxN[??a?ǒ?÷X@qVӥ????y?G??MW?"?

both?Your program is POTENTIALLY input-bound because 4.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 