# sudoku
CV algorithm to solve Sudoku from images


Part of home assignment from Skoltech course "Introduction to Computer Vision" 2020/21 AY


- sudoku.ipynb 
Main jupyter notebook with full processing of solving problem
1) Extracting mask of table sudoku from image
2) Crop and Projective transform of table sudoku
3) Extracting digits from table sudoku
4) Recognizing digits
5) Solving sudoku via this github project repositories: https://github.com/maxme1/sudoku
6) Draw digits of solved sudoku on image


- train.ipynb
File where we train 3 models
Custom dataset = its cropped images with digits of sudoku from folder with images 'data'
Defaul dataset - MNIST handwritten digits
1) Random Forest Classifier on custom dataset
Good accuracy - solving first images

2) CNN with default dataset
Bad recognition - problem with recognize digits 1 7, because comparison is very close

3) CNN with custom dataset

Good accuracy when recognize digits on train_images as RF