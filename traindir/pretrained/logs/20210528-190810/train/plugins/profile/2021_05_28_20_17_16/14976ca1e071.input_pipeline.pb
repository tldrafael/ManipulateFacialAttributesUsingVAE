	?)???pn@?)???pn@!?)???pn@	?'9????'9???!?'9???"?
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
	??aM%(@??aM%(@!??aM%(@      ??!       "	h??s?k@h??s?k@!h??s?k@*      ??!       2	?L?T????L?T???!?L?T???:	,??26 @,??26 @!,??26 @B      ??!       J	C9ѮB???C9ѮB???!C9ѮB???R      ??!       Z	C9ѮB???C9ѮB???!C9ѮB???b      ??!       JGPUY?'9???b qh???/? @y߇ʃ??V@