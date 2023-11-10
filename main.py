import cv2
import numpy as np
import time

def gaussian_kernel_generator(besarKernel, delta):
    kernelRadius = besarKernel // 2
    result = np.zeros((besarKernel, besarKernel))

    pengali = 1 / (2 * np.pi * delta ** 2)

    for filterX in range(-kernelRadius, kernelRadius+1):
        for filterY in range(-kernelRadius, kernelRadius+1):
            result[filterY + kernelRadius, filterX + kernelRadius] = \
                np.exp(-(np.sqrt(filterY ** 2 + filterX ** 2) / (delta ** 2 * 2))) * pengali
    return result

def gaussian_kernel_derivative_generator(besarKernel, delta):
    kernelRadius = besarKernel // 2
    resultX = np.zeros((besarKernel, besarKernel))
    resultY = np.zeros((besarKernel, besarKernel))

    pengali = -1 / (2 * np.pi * delta ** 4)

    for filterX in range(-kernelRadius, kernelRadius+1):
        for filterY in range(-kernelRadius, kernelRadius+1):
            resultX[filterY + kernelRadius, filterX + kernelRadius] = \
                np.exp(-(filterX ** 2 / (delta ** 2 * 2))) * pengali * filterX
            resultY[filterY + kernelRadius, filterX + kernelRadius] = \
                np.exp(-(filterY ** 2 / (delta ** 2 * 2))) * pengali * filterY
    return resultX, resultY


def harris(src, Gx, Gy, Gxy, thres, k):
    centerKernelGyGx = Gy.shape[1] // 2
    centerKernelGxy = Gxy.shape[1] // 2

    Ix2 = np.zeros((src.shape[0], src.shape[1]))
    Iy2 = np.zeros((src.shape[0], src.shape[1]))
    Ixy = np.zeros((src.shape[0], src.shape[1]))
    IR = np.zeros((src.shape[0], src.shape[1]))

    result = src.copy()

    for i in range(src.shape[1]):
        for j in range(src.shape[0]):
            sX = 0
            sY = 0

            for ik in range(-centerKernelGyGx, centerKernelGyGx + 1):
                ii = i + ik
                for jk in range(-centerKernelGyGx, centerKernelGyGx + 1):
                    jj = j + jk

                    if ii >= 0 and ii < src.shape[1] and jj >= 0 and jj < src.shape[0]:
                        sX += src[jj, ii] * Gx[centerKernelGyGx + jk, centerKernelGyGx + ik]
                        sY += src[jj, ii] * Gy[centerKernelGyGx + jk, centerKernelGyGx + ik]

            Ix2[j, i] = sX * sX
            Iy2[j, i] = sY * sY
            Ixy[j, i] = sX * sY

    for i in range(src.shape[1]):
        for j in range(src.shape[0]):
            sX2 = 0
            sY2 = 0
            sXY = 0

            for ik in range(-centerKernelGxy, centerKernelGxy + 1):
                ii = i + ik
                for jk in range(-centerKernelGxy, centerKernelGxy + 1):
                    jj = j + jk

                    if ii >= 0 and ii < src.shape[1] and jj >= 0 and jj < src.shape[0]:
                        sX2 += Ix2[jj, ii] * Gxy[centerKernelGxy + jk, centerKernelGxy + ik]
                        sY2 += Iy2[jj, ii] * Gxy[centerKernelGxy + jk, centerKernelGxy + ik]
                        sXY += Ixy[jj, ii] * Gxy[centerKernelGxy + jk, centerKernelGxy + ik]

            R = ((sX2 * sY2) - (sXY * sXY)) - k * (sX2 + sY2)**2

            if R > thres:
                IR[j, i] = R
            else:
                IR[j, i] = 0

    for y in range(1, IR.shape[0] - 1):
        for x in range(6, IR.shape[1] - 6):

            if (IR[y, x] > IR[y + 1, x] and
                IR[y, x] > IR[y - 1, x] and
                IR[y, x] > IR[y, x + 1] and
                IR[y, x] > IR[y, x - 1] and
                IR[y, x] > IR[y + 1, x + 1] and
                IR[y, x] > IR[y + 1, x - 1] and
                IR[y, x] > IR[y - 1, x + 1] and
                IR[y, x] > IR[y - 1, x - 1]):

                cv2.circle(result, (x, y), 5, 255, -1)
                result[y, x] = 255

    return result

def circleMidpoint(img, x0, y0, radius, val):
    x = radius
    y = 0
    radiusError = 1 - x

    while x > y:
        img[y + y0, x + x0] = val
        img[x + y0, y + x0] = val
        img[-y + y0, x + x0] = val
        img[-x + y0, y + x0] = val
        img[-y + y0, -x + x0] = val
        img[-x + y0, -y + x0] = val
        img[y + y0, -x + x0] = val
        img[x + y0, -y + x0] = val

        y += 1

        if radiusError < 0:
            radiusError += 2 * y + 1
        else:
            x -= 1
            radiusError += 2 * (y - x + 1)

def main():
    start = time.time()

    src = cv2.imread("/Users/mansovic./CLionProjects/CPROJECT/r.png")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    Gxy = gaussian_kernel_generator(7, 3.9)
    dGx, dGy = gaussian_kernel_derivative_generator(7, 1.3)

    print("Gxy :", Gxy)
    print("dGx :", dGx)
    print("dGy :", dGy)

    # Use the Harris function and store the result in 'corner'
    corner = harris(src, dGx, dGy, Gxy, 5000, 0.04)

    stop = time.time()
    duration = (stop - start) * 10**6  # microseconds
    print("Time taken by function: {} microseconds".format(duration))

    cv2.imshow('Original', src)
    cv2.imshow('Harris corners detected', corner)  # Now 'corner' is defined as the output of the 'harris' function.

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()