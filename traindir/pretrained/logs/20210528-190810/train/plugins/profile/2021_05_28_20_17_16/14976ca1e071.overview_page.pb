?	?)???pn@?)???pn@!?)???pn@	?'9????'9???!?'9???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?)???pn@??aM%(@1h??s?k@A?L?T???I,??26 @YC9ѮB???rEagerKernelExecute 0*	??C????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorT?n.?
2@!x@+kp?X@)T?n.?
2@1x@+kp?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetcho?UfJ???!,a?K???)o?UfJ???1,a?K???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???;?Ǯ?!??k?:??)???£??1???Fq??:Preprocessing2F
Iterator::ModelF\ ?K??!6-N????)??G????1;Չ????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap~??g2@!ұ?@b?X@)R?b??v?1D?W[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?'9???Ih???/? @Q߇ʃ??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??aM%(@??aM%(@!??aM%(@      ??!       "	h??s?k@h??s?k@!h??s?k@*      ??!       2	?L?T????L?T???!?L?T???:	,??26 @,??26 @!,??26 @B      ??!       J	C9ѮB???C9ѮB???!C9ѮB???R      ??!       Z	C9ѮB???C9ѮB???!C9ѮB???b      ??!       JGPUY?'9???b qh???/? @y߇ʃ??V@?"S
*model/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput?W????!?W????"R
)model/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput<??i???!fK ?s ??"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??9??Q??!?Ȏ??Է?0"e
9gradient_tape/model/conv2d_28/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterg??J>??! ?zc???0"y
Mgradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter??o??A??!i?+?u???0"x
Lgradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterp3?2?4??!?%s,???0"\
>gradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D?=?5??!QVD?l???0"]
?gradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DConv2D???q+??!??4?#???0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltero??W/???!?~??6???0"e
9gradient_tape/model/conv2d_29/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,???7???!`?\??0Q      Y@YxN[??a?ǒ?÷X@qMQ넦???yq0)Y?"?

both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 