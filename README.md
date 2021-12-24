# Carousel Memory: Rethinking the Design of Episodic Memory for Continual Learning

This is the source code of Carousel Memory (CarM)
![workflow](./figs/workflow_figure.png)

Continual Learning (CL) is an emerging machine learning paradigm that aims to learn from a continuous stream of tasks without forgetting knowledge learned from the previous tasks. To avoid performance decrease caused by forgetting, prior studies exploit episodic memory (EM), which stores a subset of the past observed samples while learning from new non-i.i.d. data. Despite the promising results, since CL is often assumed to execute on mobile or IoT devices, the EM size is bounded by the small hardware memory capacity and makes it infeasible to meet the accuracy requirements for real-world applications. Specifically, all prior CL methods discard samples overflowed from the EM and can never retrieve them back for subsequent training steps, incurring loss of information that would exacerbate catastrophic forgetting. We explore a novel hierarchical EM management strategy to address the forgetting issue. In particular, in mobile and IoT devices, real-time data can be stored not just in high-speed RAMs but in internal storage devices as well, which offer significantly larger capacity than the RAMs. Based on this insight, we propose to exploit the abundant storage to preserve past experiences and alleviate the forgetting by allowing CL to efficiently migrate samples between memory and storage without being interfered by the slow access speed of the storage. We call it Carousel Memory (CarM). As CarM is complementary to existing CL methods, we conduct extensive evaluations of our method with seven popular CL methods and show that CarM significantly improves the accuracy of the methods across different settings by large margins in final average accuracy while retaining the same training efficiency.

## Example
```
python main.py --config=my/config/file/path
```

## References

+ https://github.com/aimagelab/mammoth.git
+ https://github.com/drimpossible/GDumb.git
+ https://github.com/arthurdouillard/incremental_learning.pytorch.git
+ https://github.com/wuyuebupt/LargeScaleIncrementalLearning.git
+ https://github.com/zhchuu/continual-learning-reproduce.git
+ https://github.com/clovaai/rainbow-memory.git
+ https://github.com/facebookresearch/agem.git
+ https://github.com/facebookresearch/GradientEpisodicMemory.git
+ https://github.com/srebuffi/iCaRL.git
+ https://github.com/DRSAD/iCaRL.git
