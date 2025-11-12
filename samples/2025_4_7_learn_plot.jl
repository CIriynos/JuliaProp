

ADK(E) = (4 / E) * exp(- 2 / (3 * E))

a = ADK(0.05)
b = ADK(0.05 + 0.0002)
abs(a - b) / a
# 增长了5%

c = ADK(0.05)
d = ADK(0.05 + 0.00002)
abs(c - d) / c
# 增长了0.4%

xs = 0.04: 0.0001: 0.06
plot(xs, ADK.(xs))

ADK