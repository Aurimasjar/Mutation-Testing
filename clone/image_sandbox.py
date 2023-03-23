from matplotlib import pyplot as plt
import plot

# plot.plot_acc_stats([0.5, 0.4, 0.3, 0.28, 0.25, 0.21])
#
image = 'images/acc_image.png'

# img = plt.imread('C:/Users/a.petretis/Laptop backup/Universitetas/Magistro darbas/3 '
#                  'semestras/images/other/c_metrics_150/' + image)
img = plt.imread(image)
plt.imshow(img)
plt.show()
