import matplotlib.pyplot as plt
import re

loss_cls = []
loss_box_reg = []
loss_mask = []
loss_rpn_cls = []
loss_rpn_loc = []
total_loss = []

with open('./detectron2.log', 'r', encoding='utf-8') as f:
    i = 0
    for line in f:
        # pattern = r'l_objectness:\s*(\d+\.\d+),\s*l_rpn_box_reg:\s*(\d+\.\d+),\s*l_class:\s*(\d+\.\d+),\s*l_box_reg:\s*(\d+\.\d+),\s*l_mask:\s*(\d+\.\d+),\s*l_total:\s*(\d+\.\d+),'
        pattern = r'total_loss:\s*(\d+\.\d+)\s*loss_cls:\s*(\d+\.\d+)\s*loss_box_reg:\s*(\d+\.\d+)\s*loss_mask:\s*(\d+\.\d+)\s*loss_rpn_cls:\s*(\d+\.\d+)\s*loss_rpn_loc:\s*(\d+\.\d+)'
        match = re.search(pattern, line.strip())
        if match:
            loss_cls.append(float(match.group(1)))
            loss_box_reg.append(float(match.group(2)))
            loss_mask.append(float(match.group(3)))
            loss_rpn_cls.append(float(match.group(4)))
            loss_rpn_loc.append(float(match.group(5)))
            total_loss.append(float(match.group(6)))
            i += 1
print(f'Parsed {i} lines')
x = range(len(loss_cls))  # 或者用 len(total_loss)，只要长度一致

plt.figure(figsize=(10, 6))
plt.plot(x, loss_cls, label='loss_cls')
plt.plot(x, loss_box_reg, label='loss_box_reg')
plt.plot(x, loss_mask, label='loss_mask')
plt.plot(x, loss_rpn_cls, label='loss_rpn_cls')
plt.plot(x, loss_rpn_loc, label='loss_rpn_loc')
plt.plot(x, total_loss, label='total_loss', linewidth=2)

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()