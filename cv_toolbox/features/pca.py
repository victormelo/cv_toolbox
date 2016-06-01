from sklearn.decomposition import PCA
from pylab import *
from skimage import data, io, color

link = "http://perierga.gr/wp-content/uploads/2012/01/coca_cola.jpg"
coke_gray = io.imread(link,as_grey=True)

subplot(2, 2, 1)
io.imshow(coke_gray)
xlabel('Original Image')

for i in range(1, 4):
    n_comp = 5 ** i
    pca = PCA(n_components = n_comp)
    pca.fit(coke_gray)
    coke_gray_pca = pca.fit_transform(coke_gray)
# subplot(2, 2, 2)
# io.imshow(coke_gray_pca)
# xlabel('Image after applying PCA')
coke_gray_restored = pca.inverse_transform(coke_gray_pca)
subplot(2, 2, i+1)
io.imshow(coke_gray_restored)
xlabel('Restored image n_components = %s' %n_comp)
print 'Variance retained %s %%' %((1 - sum(pca.explained_variance_ratio_) / size(pca.explained_variance_ratio_)) * 100)
print 'Compression Ratio %s %%' %(float(size(coke_gray_pca)) / size(coke_gray) * 100)
show()
