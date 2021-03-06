?	h???c@h???c@!h???c@      ??!       "?
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
?| a@*      ??!       2	ђ?????ђ?????!ђ?????:	8??̒P)@8??̒P)@!8??̒P)@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?X??_?+@y??	??U@?"S
*model/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput	?wXJa??!	?wXJa??"R
)model/conv2d_transpose_5/conv2d_transposeConv2DBackpropInputl#L?{W??!??c\??"b
6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?L/W]{??!??<?ͳ?0"e
9gradient_tape/model/conv2d_28/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?N??w??!L??????0"e
9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter3???jP??!?|8`???0"-
IteratorGetNext/_4_Recv??^?[??!?^?[k??"y
Mgradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterr:?D]???!nrD0?Y??0"x
Lgradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter`??9????!dr?eD??0"\
>gradient_tape/model/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D??Kl7??!4???????0"]
?gradient_tape/model/conv2d_transpose_11/conv2d_transpose/Conv2DConv2DlĎ*Ts??!{?J_??0Q      Y@Y=??p`??a/?_<~?X@q?(???I@ydN?!Cbb?"?

both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Volta)(: B 