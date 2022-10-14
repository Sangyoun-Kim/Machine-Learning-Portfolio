#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string.h>
#include <cmath>


using namespace std;

const int MAX_LEN = 2000;

// input(X) type
using dataType = double;

// learning_rate = 0.001 from book pg. 117
constexpr double learning_rate = 1e-3;

//matrics multification
template<typename vecType>
vector<vector<vecType>> matmul(
   vector<vector<vecType>> arr1, int row1, int col1,
   vector<vector<vecType>> arr2, int row2, int col2) {

   vector<vector<vecType>> result(row1, vector<vecType>(col2));

   for (int i = 0; i < row1; ++i) {
      for (int j = 0; j < col2; ++j) {
         vecType sum = 0.0;
         for (int k = 0; k < row2; ++k) sum += arr1[i][k] * arr2[k][j];
         result[i][j] = sum;
      }
   }

   return result;
}


// swap 
void swap(double* x, double* y)
{
   double temp = *x;
   *x = *y;
   *y = temp;

}

// vector sort for range calculation and median
void quickSort(vector<double>& v, int start, int end)
{
   if (start < end) {
      int p = partition(v, start, end);
      quickSort(v, start, p - 1);
      quickSort(v, p + 1, end);
   }
}

// sum of the vector
double vectorSum(vector<double> v)
{
   int n = v.size();
   double sum = 0.0;
   for (int i = 0; i < n; i++)
      sum += v[i];

   return sum;
}

//mean of the vector
double vectorMean(vector<double> v)
{
   // Mean= sum/size
   return vectorSum(v) / v.size();
}

//median of the vector
double vectorMedian(vector<double> v)
{
   vector<double> temp = v;
   quickSort(temp, 0, v.size() - 1);
   // need to sort the vector first to be able to get middle value
   int n = temp.size();

   if (n % 2 == 0)
      //even
      return (((temp[n / 2]) + temp[(n / 2) - 1]) / 2);
   else
      //odd
      return temp[(n + 1) / 2];
}


// dividing vector for sort before range calculation
int partition(vector<double>& v, int start, int end)
{
   int s = start, piv = end;
   for (int i = start; i < end; i++)
   {
      if (v[i] < v[piv]) {
         swap(&v[i], &v[s]);
         s++;
      }
   }
   swap(&v[s], &v[piv]);
   return s;
}


// calculate the range of vector
double vectorRange(vector<double> v)
{
   vector<double> temp = v;
   int n = temp.size();

   //sort to know max - min value
   quickSort(temp, 0, n - 1);

   // return min-max value (range of vector)
   return (temp[n - 1] - temp[0]);
}

// print statistic data of vector
void print_stats(vector<double> v)
{
   cout << "Sum: " << vectorSum(v) << endl;
   cout << "Mean: " << vectorMean(v) << endl;
   cout << "Median: " << vectorMedian(v) << endl;
   cout << "Range: " << vectorRange(v) << endl;

}


// sigmoid function
double sigmoid(double z) {
   return 1.0 / (1.0 + std::exp(-z));
}

// gradient descent

std::vector<double> GradientDescent(std::vector<std::vector<dataType>>& X, std::vector<std::vector<dataType>>& Y, std::vector<std::vector<dataType>>& W, double& b, double lr) {
   const int m = Y.size() * W[0].size(); // The length of the arrangement.
   const int row1 = X.size(); // axis = 0
   const int col1 = X[0].size(); // axis = 1
   const int row2 = W.size(); // axis = 0
   const int col2 = W[0].size(); // axis = 1

   std::vector<double> lossList; // Arrangement of the loss function result.

   if (m != X.size() * W[0].size()) throw std::runtime_error("The sizes of X and Y are not right");

   // matrics multiplication
   auto matmulWX = matmul(X, row1, col1, W, row2, col2);

   // calulate dW and db
   for (int i = 0; i < row1; ++i) {
      double loss = 0, dW = 0, db = 0;

      for (int j = 0; j < col2; ++j) {
         // Hypothesis
         double H = sigmoid(matmulWX[i][j] + b);

         // sigma dW
         dW += (H - Y[i][j]) * X[i][j];

         // sigma db
         db += H - Y[i][j];
      }

      dW = dW / m; // m = row1 * col2

      db = db / m; // m = row1 * col2

      // loss function
      for (int j = 0; j < col2; ++j) {
         // hypothesis
         double H = sigmoid(matmulWX[i][j] + b);

         loss += (Y[i][j] * std::log(H) + (1 - Y[i][j]) * std::log(1 - H));
      }

      for (int i = 0; i < row2; ++i) {
         for (int j = 0; j < col2; ++j) {
            // weight and bias Update
            W[i][j] = W[i][j] - lr * dW;
            b = b - lr * db;
         }
      }

      loss = loss / -m; // m = row1 * col2

      lossList.push_back(loss);
   }

   return lossList;
}

// Logistic Regression

void LogisticRegression(std::vector<std::vector<dataType>>& X, std::vector<std::vector<dataType>>& Y, std::vector<std::vector<dataType>>& W, double& b, int epochs, double lr) {
   int printTry = epochs / 10; // Print epochs as many times as the value divided by 10.

   for (int i = 0; i < epochs; ++i) {
      auto loss = GradientDescent(X, Y, W, b, lr);
      if (i % printTry == 0) {
         std::cout << "Weight: ";
         for (int i = 0; i < W.size(); ++i) for (int j = 0; j < W[0].size(); ++j) std::cout << W[i][j] << " ";
         std::cout << "\nBias: " << b << "\nLoss: ";
         for (int j = 0; j < loss.size(); ++j) std::cout << loss[j] << ' ';
         std::cout << "\n\n";
      }
   }
}

// predicting
std::vector<std::vector<int>> predict(std::vector<std::vector<dataType>>& W, double& b, std::vector<std::vector<dataType>> X) {
   std::vector<std::vector<int>> result;

   // matrics multiplication
   auto matmulWX = matmul(X, X.size(), X[0].size(), W, W.size(), W[0].size());

   for (int i = 0; i < X.size(); ++i) {
      result.push_back(std::vector<int>());

      for (int j = 0; j < W[0].size(); ++j) {
         result[i].push_back((sigmoid(matmulWX[i][j] + b) < 0.5) ? 0 : 1);
      }
   }

   return result;
}

// function to print prediction 
void printPrediction(std::vector<std::vector<int>> prediction) {
   std::cout << "Prediction: ";
   for (int i = 0; i < prediction.size(); ++i) {
      for (int j = 0; j < prediction[i].size(); ++j) {
         std::cout << (prediction[i][j] == 0 ? "dead" : "survived") << ' ';
      }
      std::cout << '\n';
   }
}

// calculate accuracy

int main() {
   ifstream inFS;
   string line;
   string ID_in, pclass_in, survived_in, sex_in, age_in;

   vector<string> ID(MAX_LEN);
   vector<double> pclass(MAX_LEN);
   vector<double> survived(MAX_LEN);
   vector<double> sex(MAX_LEN);
   vector<double> age(MAX_LEN);



   cout << "Opening file titanic_project.csv." << endl;
   // opening data file
   inFS.open("titanic_project.csv");
   if (!inFS.is_open()) {
      cout << "Could not open file titanic_project.csv." << endl;
      return 1;
   }

   // reading in data from csv file
   cout << "Reading line 1" << endl;
   getline(inFS, line);


   // echo heading
   cout << "heading: " << line << endl;

   int numObservations = 0;
   while (inFS.good()) {
      getline(inFS, ID_in, ',');
      getline(inFS, pclass_in, ',');
      getline(inFS, survived_in, ',');
      getline(inFS, sex_in, ',');
      getline(inFS, age_in, '\n');

      ID.at(numObservations) = ID_in;
      pclass.at(numObservations) = stof(pclass_in);
      survived.at(numObservations) = stof(survived_in);
      sex.at(numObservations) = stof(sex_in);
      age.at(numObservations) = stof(age_in);

      numObservations++;
   }

   ID.resize(numObservations);
   pclass.resize(numObservations);
   survived.resize(numObservations);
   sex.resize(numObservations);
   age.resize(numObservations);

   cout << "new length " << pclass.size() << endl;

   cout << "Closing file titanic_project.csv." << endl;
   inFS.close();// Done with file, so close it

   cout << "Number of records: " << numObservations << endl;

   cout << "\nStats for pclass" << endl;
   print_stats(pclass);

   cout << "\nStats for survived" << endl;
   print_stats(survived);

   cout << "\nStats for sex" << endl;
   print_stats(sex);

   cout << "\nStats for age" << endl;
   print_stats(age);

   //split vector into train-test

   //read first 800 data for train
   vector<double> pclass_train = vectorSplit(pclass, 0, 800);
   vector<double> survived_train = vectorSplit(survived, 0, 800);
   vector<double> sex_train = vectorSplit(sex, 0, 800);
   vector<double> age_train = vectorSplit(age, 0, 800);

   // create mother train vector with child vectors  
   vector<vector<double>> train = { pclass_train, survived_train, sex_train, age_train };

   //print train stats
   cout << "\nStats for pclass_train" << endl;
   print_stats(pclass_train);

   cout << "\nStats for survived_train" << endl;
   print_stats(survived_train);

   cout << "\nStats for sex_train" << endl;
   print_stats(sex_train);

   cout << "\nStats for age_train" << endl;
   print_stats(age_train);


   // test vectors
   vector<double> pclass_test = vectorSplit(pclass, 801, pclass.size() - 1);
   vector<double> survived_test = vectorSplit(survived, 801, survived.size() - 1);
   vector<double> sex_test = vectorSplit(sex, 801, sex.size() - 1);
   vector<double> age_test = vectorSplit(age, 801, age.size() - 1);

   // create mother test vector with child test vectors
   vector<vector<double>> test = { pclass_test, survived_test, sex_test, age_test };

   //print test stat
   cout << "Stats for pclass_test" << endl << endl;
   print_stats(pclass_test);

   cout << "Stats for survived_test" << endl << endl;
   print_stats(survived_test);

   cout << "Stats for sex_test" << endl << endl;
   print_stats(sex_test);

   cout << "Stats for age_test" << endl << endl;
   print_stats(age_test);

   cout << endl << endl;

   /////////
   //Logistic regression




   return 0;
}