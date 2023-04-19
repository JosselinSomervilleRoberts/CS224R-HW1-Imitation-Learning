lr = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
avg_return = [-283.47, -538.49, -809.70, -737.98, 3282.81, 4669.14, 985.69, 236.37, -44798.90]

# Plot in semilogx scale
import matplotlib.pyplot as plt
plt.semilogx(lr, avg_return, 'o-')
plt.xlabel('Learning Rate')
plt.ylabel('Average Return')
plt.title('Learning Rate vs. Average Return')
plt.ylim(-1000, 5000)
plt.show()