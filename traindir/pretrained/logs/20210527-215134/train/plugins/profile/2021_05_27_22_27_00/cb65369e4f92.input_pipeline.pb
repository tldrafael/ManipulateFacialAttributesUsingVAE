	??
?gn@??
?gn@!??
?gn@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??
?gn@r?#l(@1???y?k@A?R)v4??IZ~?*Op@rEagerKernelExecute 0*	???a??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorw????I,@!?z??q?X@)w????I,@1?z??q?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?9?S?ɢ?!c?K?<~??)?9?S?ɢ?1c?K?<~??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??Χ???!??????)8L4H?S??1?<?????:Preprocessing2F
Iterator::ModelGɫsȶ?!h?f????)np?????1??jOX??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapM?O?L,@!??2C ?X@)??&OYMw?1*???t??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?21??? @Q??YG"?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	r?#l(@r?#l(@!r?#l(@      ??!       "	???y?k@???y?k@!???y?k@*      ??!       2	?R)v4???R)v4??!?R)v4??:	Z~?*Op@Z~?*Op@!Z~?*Op@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?21??? @y??YG"?V@