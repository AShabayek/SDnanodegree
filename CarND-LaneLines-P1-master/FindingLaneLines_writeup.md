#**Finding Lane Lines on the Road** 


---

**Finding Lane Lines Using Canny edge detection and hough transform**

Project objectives:
* Detect lane markings on the road and have them presented as lines.
* Extrapolate an averaged line representing all relevant lane markings for each side.
* Explain and reflect upon method and approaches used.


[//]: # (Image References)

[image1]: LDsolidWhiteRight.jpg "Grayscale"
[image2]: LDsolidYellowCurve.jpg "Grayscale"
[image3]: LDsolidYellowLeft.jpg "Grayscale"
[image4]: LDwhiteCarLaneSwitch.jpg "Grayscale"

---

### Reflection

###1. Pipline and parameter selection

The lane detection pipline consists of 5 main stages from filtering and masking to edge and line detection. The stages are the following respectively :
          1.Image to grayscale.
          2.Improve image image quality by removing blured region by a mean/gaussian filter.
          3.Create an empty copy of the image to use as a mask for the canny edge detection vertices outputtted and merge it with a bitwise and with the input img .
          4.Apply the hough transform on the masked region of interset to get all the vertices forming a line.
          5.Lastly is the draw_lines function where the filtering of detected lines, averaging and extrapolation happens.
##Line filtering
In draw_lines there are 3 filtering stages. first is based on the image x axis lenght. First half is classified for the left lane lines, second is for right lane line. there is also a slight filtering in the y-axis to avoid and trim the noise lines at image corners.
>if (((imagemidpointx) < x1 ) and ((imagemidpointx) < x2 ) and((imagemidpointy+80) < y2) and ((imagemidpointy+80) < y1))

Afterwards filtering based on slope left lanes where a set angle range from 20 to 40 degrees for the right lane line and -20 to -40 for the left lane line is used. The ranges where concluded using both observation and expermenting. this filtering is done only on the first run as afterwards averaging is the main decision maker as will be discussed later.
>if   20<(np.degrees(np.arctan((y2-y1)/(x2-x1)))) < 40:

>if   -20>(np.degrees(np.arctan((y2-y1)/(x2-x1)))) > -40:

Lastly is the filtering based on the difference of the averaged slope from previous detected lines and the current processed.

>if np.absolute(np.degrees(np.arctan((y2-y1)/(x2-x1)- np.mean(lineaverage)))) < 30  :

## Averaging Lines:
Averaging occurs on both lines per frame and on every incoming frame with regards to previous frames.
First averaging on lines per frame:
>if np.absolute(np.degrees(np.arctan((y2-y1)/(x2-x1)- np.mean(lineaverage)))) < 30  :

This averaging check for the angle difference between the average of previous lines against the current one in iteration, this check however is only triggered after the first line is processed based on the the intial angle filtering.

Second is the averaging of frames. In this stage averaging of previous extrapolated lines happen in order to reduce both error in gradient and in line position.
>if      (0 < lineleft[2] < img.shape[1] ) and (0 < lineleft[3] < img.shape[0]):
         averagelineleftx.append(lineleft[0])
         averagelinelefty.append(lineleft[1])
         averageleftx.append(lineleft[2])
         averagelefty.append(lineleft[3])
         Line2x1 = math.floor(np.mean(averageleftx)-1000* np.mean(averagelineleftx))
    Line2y1 = math.floor(np.mean(averagelefty)-1000*np.mean(averagelinelefty))
    Line2x2 = math.floor(np.mean(averageleftx)+100*np.mean(averagelineleftx))
    Line2y2 = math.floor(np.mean(averagelefty)+100*np.mean(averagelinelefty))

Here is also where part of the extrapolation occurs which takes us to the last phase ,extrapolation.

#Extrapolation

in the extrapolation step after using the colinear vector to determine the line direction and the coordinates  returned back from opencv fitline function the straight line equation is modelled and used to calculate the y point where x value is close to half of the image as shown below :

>    slope = (Line2y2 - Line2y1) / (Line2x2 - Line2x1)
    c = Line2y2 - (slope * Line2x2 )
    y = ((imagemidpointx)-40)* slope + c
    Line1x1 = math.floor(imagemidpointx)+40
    Line1y1 = math.floor(y)
    
#Image example results
![alt text-1][image1] ![alt text-2][image2]![alt text-2][image3]


###2.  shortcomings of pipeline

Most of this pipline problems lie in the filtering which only shows in the challange example video. first lies in the  y axis border filtering which is image size dependent and not fine tuned as tolerances as the 80 pixels is highly dependent on the image size as well as camera focus. Also the x axis filtering of left and right lane line while its functional is highly dependent on the image symetery assuming camera is centered.

Last but not least the averaging between frames doesnt include a filter for frames with alot of noise and badly estimated values thus assuming past frame output is correct given the simple extreme value validty check. this reduces the accuracy significantly.


###3.Further Improvements to your pipeline

parameters such as the 40 or 80 pixels used in the filtering and extrapolation should be image size relative. Also further filtering between frames should be included for as mentioned they reduce accuracy. Line filtering should not be based on the angle range determined from observations even if only used in the first iteration only. An intial wide tolerance should be assigned that is reduced over the different  line iterations thus allowing flexible angle ranges while creating a filter based on average over the lines thus far.
Finally for curve roads extrapolation should be implemented diffferently accounting for change occuring, this should be done by taking into account the lines with changing gradient and extrapolating only to their starting point not till mid image border.
