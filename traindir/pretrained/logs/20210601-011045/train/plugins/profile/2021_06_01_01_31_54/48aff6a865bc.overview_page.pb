?	]?C?mc@]?C?mc@!]?C?mc@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC]?C?mc@O"¿?!@1?[??`@A????+??I?+?V]/&@rEagerKernelExecute 0*	?Q?Eu?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???@??)@!??I???X@)???@??)@1??I???X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismsL????!^??8????)??qo~ì?1?]uJ????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??~?n??!???Mz???)??~?n??1???Mz???:Preprocessing2F
Iterator::Model,??̸?!?KC`???)"??u????1??MR????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapl$	??)@!i?y?o?X@)_%??t?1W?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI(?Ң?8*@Q{?????U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	O"¿?!@O"¿?!@!O"¿?!@      ??!       "	?[??`@?[??`@!?[??`@*      ??!       2	????+??????+??!????+??:	?+?V]/&@?+?V]/&@!?+?V]/&@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q(?Ң?8*@y{?????U@?"S
*model/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput????K???!????K???"R
)model/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput?5"???!??V1???"b
6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter+???*Ж?!???\#???0"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??d???!(??C(???0"e
9gradient_tape/model/conv2d_28/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?? Ք???!?2y????0"x
Lgradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilteryx?
?Վ?!q?G?"???0"y
Mgradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter?z?tĐ??!~?D/???0"]
?gradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DConv2DH?\?_9??!?F=?2??0"\
>gradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D@???%??!?@?????0"-
IteratorGetNext/_4_Recv???5???!6? {7??Q      Y@Y=??p`??a/?_<~?X@q?yu????y0I-??]i?"?

both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Volta)(: B 