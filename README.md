# DIP_A2

Question 1.

Write image processing techniques to determine the current time shown by the clock.
Hint: Try to segment the hour and minute hands and determine the angles between them. This
will help you determine the time.

Question 2.

The following image consists of two sections. The upper portion contains the character you
wish to use as input, while the lower section is designated for conducting a search. You should
write a code that can identify a matching character in the lower section when different characters
are displayed one at a time in the upper section. Once a match is found, the code should draw a
green bounding box around the corresponding matched character box(es). In the event that no
match is discovered, the code should display the message "No match found."

Question 3.

The test scores for reading, writing and speaking for a student are given below. The score is between
the scale 0 – 10. The blue bars indicated the score for each test for a particular student. For your
reference scores, the actual scores indicated by bars are also shown in the squares at the end of each
bar.
Write a program to determine the score of a student in each test using the bars shown and display the
scores in each test on the image itself.

Question 4.

Create a comprehensive image processing software with a user-friendly interface that offers various
functionalities and presents different outcomes for diverse operations. Your task is to design software
that implements the following functionalities:
- Enable image loading, display and the option to save the output image.
- Adjust image brightness.
- Execute various Log operations on images, allowing users to choose parameter values from the
user interface (UI).
- Perform thresholding for both color and grayscale images.
- Implement color conversion capabilities.
- Allow users to draw shapes on the image using the mouse.
- Apply blurring using different filters, granting users the ability to select different filters.
- Perform image sharpening using Laplacian techniques.
- Implement Unsharp Masking.
- Enable color-based image segmentation.
- Detect lines using Hough Transform.
- Perform Morphological Operations, including Erode, Dilate, Open, and Close.
- Execute Connected Component Analysis.
- Detect contours, including area, smallest object, largest object, and hole detection. The software
should display object details upon clicking with the mouse, such as area and parameters.
- Determine time: call the function that you developed in question 1
- Match Input: Call the function you developed in question 2.
- Score Calculation: call the function that you developed in question 3.
Additionally, package your software using a Docker container to ensure straightforward execution on
any system without complications.
