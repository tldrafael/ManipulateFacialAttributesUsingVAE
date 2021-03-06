?	|,}?B?t@|,}?B?t@!|,}?B?t@	J1?c?'??J1?c?'??!J1?c?'??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL|,}?B?t@???Y@10o???k@Aͬ??????I?T?]?@Y?yrM????rEagerKernelExecute 0*	D?l?k+?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Y?w"@!?d?W??X@)?Y?w"@1?d?W??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?d??~???!?H?H??)???R???1vț|????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch1AG?Z??!?YA|???)1AG?Z??1?YA|???:Preprocessing2F
Iterator::Model|DL?$z??!4?L'???)???,?~?1??C?n???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?XP??"@!?f?[??X@)hY????p?1v=? ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 30.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9I1?c?'??I6??@@QYw??P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Y@???Y@!???Y@      ??!       "	0o???k@0o???k@!0o???k@*      ??!       2	ͬ??????ͬ??????!ͬ??????:	?T?]?@?T?]?@!?T?]?@B      ??!       J	?yrM?????yrM????!?yrM????R      ??!       Z	?yrM?????yrM????!?yrM????b      ??!       JGPUYI1?c?'??b q6??@@yYw??P@?"S
*model/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput>??JO??!>??JO??"R
)model/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput??s???!???^???"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ݶ?hb??!????G???0"e
9gradient_tape/model/conv2d_28/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?WUw?\??!??i?k???0"y
Mgradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterE^<??R??!?|?????0"x
Lgradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFiltercJ???C??!f?????0"\
>gradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D????H??!-O?????0"]
?gradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DConv2D/?͖?;??!?Z??????0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterqGԌ?!#?g????0"e
9gradient_tape/model/conv2d_29/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?)??ǌ?!4???s??0Q      Y@Y??}ylE??a?N??X@qw?ƺ.@y?9v(qUW?"?

both?Your program is POTENTIALLY input-bound because 30.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?15.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 