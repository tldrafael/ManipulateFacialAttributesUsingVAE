	h???c@h???c@!h???c@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCh???c@/???u?"@1??
?| a@Aђ?????I8??̒P)@rEagerKernelExecute 0*	?5^??I?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator'L?ʆ @!??ӽ??X@)'L?ʆ @1??ӽ??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism'?_[????!/?ޡ????)%Z?xZ~??1/??????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???????!/? ?:??)???????1/? ?:??:Preprocessing2F
Iterator::Model???6p??!:?[?.??)?|?H?F??1P???He??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???B?? @!H????X@)`=?[?w?1B?$??K??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?X??_?+@Q??	??U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/???u?"@/???u?"@!/???u?"@      ??!       "	??
?| a@??
?| a@!??
?| a@*      ??!       2	ђ?????ђ?????!ђ?????:	8??̒P)@8??̒P)@!8??̒P)@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?X??_?+@y??	??U@