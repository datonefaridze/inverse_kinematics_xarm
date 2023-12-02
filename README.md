# inverse_kinematics_xarm

The project is intended as research project (therefore it needs refactoring, i will refactor it if somebody takes interest in it).
In this project I demonstrate Xarm's ability to pick up an object. This problem is generally hard, because classical **RL** algorithms fail, because the sequence of steps that robot has to take before it reaches the goal is very large,
therefore they don't get reward in a short term, that's why they fail. In this project I used imitation learning which basically mimics the expert policy (like babies mimic adults and learn). </br>
Before this project I used lot of approaches (I worked on it throught a year) but all of them failed (I believe that was because of lack of data, even though I had in total 2 hours of training samples, i believe it should have been at least 9).
</br>
At the end I tried [eai-vc](https://github.com/facebookresearch/eai-vc) (Meta project). They basically trained backbone network  on large set of tasks, their intention was to develop general control network which could be fine tunned on specific task (like models on imagenet).
I took their model, used it as feature extractor and then fine tunned with behaviour clonning on my specific dataset.
