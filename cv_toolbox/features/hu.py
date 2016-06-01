# encoding: utf-8

import numpy as np
import Image
import matplotlib.pyplot as plt


def hu_moments(img):
    h, w = img.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    f = img

    m = dict()

    m["00"] = f.flatten().sum()

    # in case m00 is zero
    if m["00"] == 0:
        m["00"] = np.finfo(float).eps

    m["10"] = np.sum(x * f)
    m["01"] = np.sum(y * f)
    m["11"] = np.sum(x * y * f)
    m["20"] = np.sum(x**2 * f)
    m["02"] = np.sum(y**2 * f)
    m["30"] = np.sum(x**3 * f)
    m["03"] = np.sum(y**3 * f)
    m["12"] = np.sum(x * y**2 * f)
    m["21"] = np.sum(x**2 * y * f)

    xcenter = m['10'] / m['00']
    ycenter = m['01'] / m['00']
    eta = dict()

    eta['11'] = (m['11'] - ycenter * m['10']) / m['00']**2
    eta['20'] = (m['20'] - xcenter * m['10']) / m['00']**2
    eta['02'] = (m['02'] - ycenter * m['01']) / m['00']**2

    eta['30'] = (m['30'] - (3 * xcenter * m['20']) +
                 (2 * xcenter**2) * m['10']) / (m['00']**2.5)

    eta['03'] = (
        m['03'] - 3 * ycenter * m['02'] + 2 * ycenter**2 * m['01']) / m['00']**2.5
    eta['21'] = (m['21'] - 2 * xcenter * m['11'] - ycenter *
                 m['20'] + 2 * xcenter**2 * m['01']) / m['00']**2.5
    eta['12'] = (m['12'] - 2 * ycenter * m['11'] - xcenter *
                 m['02'] + 2 * ycenter**2 * m['10']) / m['00']**2.5

    phi = dict()

    phi['1'] = eta['20'] + eta['02']
    phi['2'] = (eta['20'] - eta['02'])**2 + 4 * eta['11']**2
    phi['3'] = (eta['30'] - 3 * eta['12'])**2 + (3 * eta['21'] - eta['03'])**2

    phi['4'] = (eta['30'] + eta['12'])**2 + (eta['21'] + eta['03'])**2

    phi['5'] = (eta['30'] - 3 * eta['12']) * (eta['30'] + eta['12']) * \
               ( (eta['30'] + eta['12'])**2 - 3 * (eta['21'] + eta['03'])**2 ) + \
               (3 * eta['21'] - eta['03']) * (eta['21'] + eta['03']) * \
               (3 * (eta['30'] + eta['12'])**2 - (eta['21'] + eta['03'])**2)

    phi['6'] = (eta['20'] - eta['02']) * ((eta['30'] + eta['12'])**2 -
                                          (eta['21'] + eta['03'])**2 ) + \
        4 * eta['11'] * (eta['30'] + eta['12']) * (eta['21'] + eta['03'])
    phi['7'] = (3 * eta['21'] - eta['03']) * (eta['30'] + eta['12']) * \
               ( (eta['30'] + eta['12'])**2 - 3 * (eta['21'] + eta['03'])**2) + \
               (3 * eta['12'] - eta['30']) * (eta['21'] + eta['03']) * \
               (3 * (eta['30'] + eta['12'])**2 - (eta['21'] + eta['03'])**2)

    return m, eta, phi


if __name__ == '__main__':
    img = np.array(Image.open('../images/fig1137-original.tif').convert('L'))
    img2 = np.array(Image.open('../images/fig1137-45.tif').convert('L'))
    img3 = np.array(Image.open('../images/fig1137-mirror.tif').convert('L'))
    img4 = np.array(
        Image.open('../images/fig1137-halfpadded.tif').convert('L'))
    img5 = np.array(Image.open('../images/fig1137-padded.tif').convert('L'))
    img6 = np.array(Image.open('../images/fig1137-90.tif').convert('L'))

    m, eta, phi = hu_moments(img)
    m2, eta2, phi2 = hu_moments(img2)
    m3, eta3, phi3 = hu_moments(img3)
    m4, eta4, phi4 = hu_moments(img4)
    m5, eta5, phi5 = hu_moments(img5)
    m6, eta6, phi6 = hu_moments(img6)

    print 'Original image\n %.4g %.4g %.4g %.4g %.4g %.4g %.4g' % \
        (phi['1'], float(phi['2']), phi['3'],
         phi['4'], phi['5'], phi['6'], phi['7'])

    print 'Rotated 45\n %.4g %.4g %.4g %.4g %.4g %.4g %.4g' % \
        (phi2['1'], phi2['2'], phi2['3'], phi2[
         '4'], phi2['5'], phi2['6'], phi2['7'])

    print 'Mirrored\n %.4g %.4g %.4g %.4g %.4g %.4g %.4g' % \
        (phi3['1'], phi3['2'], phi3['3'], phi3[
         '4'], phi3['5'], phi3['6'], phi3['7'])

    print 'Half padded\n %.4g %.4g %.4g %.4g %.4g %.4g %.4g' % \
        (phi4['1'], phi4['2'], phi4['3'], phi4[
         '4'], phi4['5'], phi4['6'], phi4['7'])

    print 'Padded\n %.4g %.4g %.4g %.4g %.4g %.4g %.4g' % \
        (phi5['1'], phi5['2'], phi5['3'], phi5[
         '4'], phi5['5'], phi5['6'], phi5['7'])

    print 'Rotated 90\n %.4g %.4g %.4g %.4g %.4g %.4g %.4g' % \
        (phi6['1'], phi6['2'], phi6['3'], phi6[
         '4'], phi6['5'], phi6['6'], phi6['7'])

    ax = plt.subplot('231')
    ax.set_title("Original image")
    ax.imshow(img, 'gray')
    ax.axis('off')

    ax = plt.subplot('232')
    ax.set_title("Rotated 45")
    ax.imshow(img2, 'gray')
    ax.axis('off')

    ax = plt.subplot('233')
    ax.set_title("Mirrored")
    ax.imshow(img3, 'gray')
    ax.axis('off')

    ax = plt.subplot('234')
    ax.set_title("Half padded")
    ax.imshow(img4, 'gray')
    ax.axis('off')
    ax = plt.subplot('235')
    ax.set_title("Padded")
    ax.imshow(img5, 'gray')
    ax.axis('off')

    ax = plt.subplot('236')
    ax.set_title("Rotated 90")
    ax.imshow(img6, 'gray')
    ax.axis('off')

    plt.show()

    debug = False

    if debug:
        import cv2
        cv2moments = cv2.moments(img)
        print "m10", m["10"] == cv2moments["m10"]
        print "m01", m["01"] == cv2moments["m01"]
        print "m11", m["11"] == cv2moments["m11"]
        print "m20", m["20"] == cv2moments["m20"]
        print "m02", m["02"] == cv2moments["m02"]
        print "m30", m["30"] == cv2moments["m30"]
        print "m03", m["03"] == cv2moments["m03"]
        print "m12", m["12"] == cv2moments["m12"]
        print "m21", m["21"] == cv2moments["m21"]

        print 'nu11', eta['11'], cv2moments['nu11']
        print 'nu20', eta['20'], cv2moments['nu20']
        print 'nu02', eta['02'], cv2moments['nu02']
        print 'nu30', eta['30'], cv2moments['nu30']
        print 'nu03', eta['03'], cv2moments['nu03']
        print 'nu21', eta['21'], cv2moments['nu21']
        print 'nu12', eta['12'], cv2moments['nu12']
        cv2hu = cv2.HuMoments(cv2moments).flatten()
        print 'phi1', phi['1'], cv2hu[0]
        print 'phi2', phi['2'], cv2hu[1]
        print 'phi3', phi['3'], cv2hu[2]
        print 'phi4', phi['4'], cv2hu[3]
        print 'phi5', phi['5'], cv2hu[4]
        print 'phi6', phi['6'], cv2hu[5]
        print 'phi7', phi['7'], cv2hu[6]
