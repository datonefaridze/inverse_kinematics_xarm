# 🤖 inverse_kinematics_xarm


youtube link: https://www.youtube.com/watch?v=TReSMKwKTtM&ab_channel=datonefaridze
![image](https://github.com/datonefaridze/inverse_kinematics_xarm/assets/52849166/77c862dd-84a9-4d7f-b65c-7154035cb60d)



🔍 **Project Overview**

This is research project aimed at overcoming a challenging problem in robotics—teaching an Xarm to pick up objects. Traditional Reinforcement Learning (RL) methods often fail due to the extensive sequence of actions required, which delays rewards and hinders learning.

👶 **Imitation Learning**

This project uses imitation learning algorithm to learn the expert policy (I created controller which helped me to collect the data). The process is similar to babies learning from adults. 

💡 **Previous Attempts**

Prior to this breakthrough, I explored numerous strategies throught a year (**Yeah I have devouted countless hours to this**), none of worked, because of insufficient data (at least that's my prediction). My hypothesis was that a minimum of 9 hours was essential, whereas only 2 hours were available.

🌐 **Integration with [eai-vc](https://github.com/facebookresearch/eai-vc)**

In a quest for a solution, I integrated the [eai-vc](https://github.com/facebookresearch/eai-vc) model, developed by Meta's research team. This model, trained on a vast array of tasks, aims to create a versatile control network adaptable to specific tasks, similar to models trained on ImageNet. </br>
By using it as a feature extractor, I was able to fine-tune it with behavior cloning on a specialized dataset, tailoring it to meet the unique demands of the Xarm project.

✨ **Interested in Contributing?**

The project is open for refactoring and further development. Your contributions and interest can help take this research to the next level!

---

Feel free to reach out if you're interested in collaborating or have any suggestions!

[![GitHub stars](https://img.shields.io/github/stars/datonefaridze/inverse_kinematics_xarm.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/datonefaridze/inverse_kinematics_xarm/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/datonefaridze/inverse_kinematics_xarm.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/datonefaridze/inverse_kinematics_xarm/network/)
[![GitHub watchers](https://img.shields.io/github/watchers/datonefaridze/inverse_kinematics_xarm.svg?style=social&label=Watch&maxAge=2592000)](https://GitHub.com/datonefaridze/inverse_kinematics_xarm/watchers/)
