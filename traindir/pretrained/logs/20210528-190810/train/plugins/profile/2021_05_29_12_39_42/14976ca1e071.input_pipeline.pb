	|,}?B?t@|,}?B?t@!|,}?B?t@	J1?c?'??J1?c?'??!J1?c?'??"?
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
	???Y@???Y@!???Y@      ??!       "	0o???k@0o???k@!0o???k@*      ??!       2	ͬ??????ͬ??????!ͬ??????:	?T?]?@?T?]?@!?T?]?@B      ??!       J	?yrM?????yrM????!?yrM????R      ??!       Z	?yrM?????yrM????!?yrM????b      ??!       JGPUYI1?c?'??b q6??@@yYw??P@