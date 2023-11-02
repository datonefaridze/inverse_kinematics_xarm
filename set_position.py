import xarm
import time

# id=4  მარჯვნივ
# id=5 მარცხნივ
# id=3 მარცხნივ


arm = xarm.Controller('USB')

# ათვლა იწყება პირველი ელემენტიდან ანუ [40, 0, 0, 0, 0, 0] მიანიჭებს id=1 ს 40 გრადუსს
reset = [40, 0, 0, 0, 0, 0]
add = 0
reset = [x+add for x in reset]
print(reset)
reset_id = [[x+1, float(reset[x])] for x in range(len(reset))]
print(reset_id)

arm.setPosition(reset_id, duration=1000, wait=True)

# for i in range(1, 7):
#     # arm.setPosition(i, float(reset[i]), wait=True)
#     # arm.setPosition(reset_id, wait=True)

#     # position = arm.getPosition(i, True)
#     print(i, float(reset[i]))
#     position = arm.getPosition(i, True)
#     print('position', position)


