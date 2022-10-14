

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <string>

# define M_PI           3.14

using namespace std;

const int MAX_LEN = 2000;



double calc_age_lh(double v, double mean_v, double var_v)
{
   return (1 / sqrt(2 * M_PI * var_v) * exp(-(pow(v - mean_v, 2)) / (2 * var_v)));
}


vector<double> raw_prob(int pclass, int sex, double age, double lh_pclass[2][3], double lh_sex[2][2], vector<double> ap, vector<double> age_mean, vector<double> age_var)
{
   double num_s = lh_pclass[1][pclass - 1] * lh_sex[1][sex - 1] * ap[1] *
      calc_age_lh(age, age_mean[1], age_var[1]);
   double num_p = lh_pclass[0][pclass - 1] * lh_sex[0][sex - 1] * ap[0] *
      calc_age_lh(age, age_mean[0], age_var[0]);

   double denom = lh_pclass[1][pclass - 1] * lh_sex[1][sex - 1] *
      calc_age_lh(age, age_mean[1], age_var[1]) * ap[1] +
      lh_pclass[0][pclass - 1] * lh_sex[0][sex - 1] *
      calc_age_lh(age, age_mean[0], age_var[0]) * ap[0];

   vector<double> result(2);
   result[0] = num_s / denom;
   result[1] = num_p / denom;
   return result;
}

double mean_condition(vector<int> x, vector<int> y, int condition)
{
   double sum = 0;
   int n = x.size();
   for (int i = 0; i < n; i++)
      if (y[i] == condition)
         sum += x[i];

   return sum / (double)n;
}

double var_condition(vector<int> x, vector<int> y, int con)
{
   double mean = mean_condition(x, y, con);
   double var = 0;
   int n = x.size();

   for (int i = 0; i < n; i++)
   {
      var += (x[i] - mean) * (x[i] - mean);
   }

   return var / (double)(n - 1);
}

// prior probability
vector<double> prior(vector<int> v)
{
   vector<double> ap(2);
   ap[0] = count(v.begin(), v.end(), 0) / (double)v.size();
   ap[1] = count(v.begin(), v.end(), 1) / (double)v.size();

   return ap;
}

//nth row of the vector
int nrow(vector<int> v, double target1, vector<int> y, double target2)
{
   int n = v.size();
   int count = 0;
   for (int i = 0; i < n; i++)
      if (v[i] == target1 && y[i] == target2)
         count++;
   return count;
}

//put data into the other vector as given number
vector<int> vectorSplit(vector<int> v, int l, int r)
{
   int n = v.size();

   vector<int> x(MAX_LEN);
   x.resize(r - l);
   int j = 0;
   for (int i = l; i < r; i++)
   {
      x[j] = v[i];
      j++;
   }

   return x;
}



// swap 
void swap(int* x, int* y)
{
   double temp = *x;
   *x = *y;
   *y = temp;

}

// dividing vector for sort before range calculation
int partition(vector<int>& v, int start, int end)
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

// vector sort for range calculation and median
void quickSort(vector<int>& v, int start, int end)
{
   if (start < end) {
      int p = partition(v, start, end);
      quickSort(v, start, p - 1);
      quickSort(v, p + 1, end);
   }
}

// sum of the vector
double vectorSum(vector<int> v)
{
   int n = v.size();
   double sum = 0.0;
   for (int i = 0; i < n; i++)
      sum += v[i];

   return sum;
}

//mean of the vector
double vectorMean(vector<int> v)
{
   // Mean= sum/size
   return vectorSum(v) / v.size();
}

//median of the vector
double vectorMedian(vector<int> v)
{
   vector<int> temp = v;
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



// calculate the range of vector
double vectorRange(vector<int> v)
{
   vector<int> temp = v;
   int n = temp.size();

   //sort to know max - min value
   quickSort(temp, 0, n - 1);

   // return min-max value (range of vector)
   return (temp[n - 1] - temp[0]);
}

// print statistic data of vector
void print_stats(vector<int> v)
{
   cout << "Sum: " << vectorSum(v) << endl;
   cout << "Mean: " << vectorMean(v) << endl;
   cout << "Median: " << vectorMedian(v) << endl;
   cout << "Range: " << vectorRange(v) << endl;

}


double calc_acc(vector<int> v, vector<double> prediction) {
   int wrong = 0;
   int correct = 0;
   for (int i = 0; i < v.size(); i++) {
      if (prediction[i] > 0.5 && v[i] == 1)
         correct++;
      else if (prediction[i] > 0.5 && v[i] == 0)
         wrong++;
      else if (prediction[i] < 0.5 && v[i] == 1)
         wrong++;
      else
         correct++;
   }

   return correct / (wrong + correct);
}

double calc_spec(vector<int> v, vector<double> prediction) {
   int wrong = 0;
   int correct = 0;
   int trueNeg = 0, falsePos = 0;
   for (int i = 0; i < v.size(); i++) {
      if (prediction[i] > 0.5 && v[i] == 1) {
         correct++;
      }
      else if (prediction[i] > 0.5 && v[i] == 0) {
         wrong++;
         falsePos++;
      }
      else if (prediction[i] < 0.5 && v[i] == 1) {
         wrong++;
      }
      else {
         correct++;
         trueNeg++;
      }
   }

   return trueNeg / (trueNeg + falsePos);
}

double calc_sens(vector<int> v, vector<double> prediction) {
   int wrong = 0;
   int correct = 0;
   int truePos = 0, falseNeg = 0;
   for (int i = 0; i < v.size(); i++) {
      if (prediction[i] > 0.5 && v[i] == 1) {
         correct++;
         truePos++;
      }
      else if (prediction[i] > 0.5 && v[i] == 0) {
         wrong++;

      }
      else if (prediction[i] < 0.5 && v[i] == 1) {
         wrong++;
         falseNeg++;
      }
      else {
         correct++;

      }
   }

   return truePos / (truePos + falseNeg);
}



int main()
{
   ifstream inFS;
   string line;
   string ID_in, pclass_in, survived_in, sex_in, age_in;

   vector<string> ID(MAX_LEN);
   vector<int> pclass(MAX_LEN);
   vector<int> survived(MAX_LEN);
   vector<int> sex(MAX_LEN);
   vector<int> age(MAX_LEN);



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

   cout << "Stats for pclass" << endl;
   print_stats(pclass);
   cout << endl;

   cout << "Stats for survived" << endl;
   print_stats(survived);
   cout << endl;

   cout << "Stats for sex" << endl;
   print_stats(sex);
   cout << endl;

   cout << "Stats for age" << endl;
   print_stats(age);
   cout << endl;


   //split vector into train-test

   cout << "Split data into train-test" << endl;
   //read first 800 data for train
   vector<int> pclass_train = vectorSplit(pclass, 0, 800);
   vector<int> survived_train = vectorSplit(survived, 0, 800);
   vector<int> sex_train = vectorSplit(sex, 0, 800);
   vector<int> age_train = vectorSplit(age, 0, 800);

   // create mother train vector with child vectors  
   vector<vector<int>> train = { pclass_train, survived_train, sex_train, age_train };

   //print train stats
   cout << "Stats for pclass_train" << endl;
   print_stats(pclass_train);
   cout << endl;

   cout << "Stats for survived_train" << endl;
   print_stats(survived_train);
   cout << endl;

   cout << "Stats for sex_train" << endl;
   print_stats(sex_train);
   cout << endl;

   cout << "Stats for age_train" << endl;
   print_stats(age_train);
   cout << endl;


   // test vectors
   vector<int> pclass_test = vectorSplit(pclass, 801, pclass.size() - 1);
   vector<int> survived_test = vectorSplit(survived, 801, survived.size() - 1);
   vector<int> sex_test = vectorSplit(sex, 801, sex.size() - 1);
   vector<int> age_test = vectorSplit(age, 801, age.size() - 1);

   // create mother test vector with child test vectors
   vector<vector<int>> test = { pclass_test, survived_test, sex_test, age_test };

   //print test stat
   cout << "Stats for pclass_test" << endl;
   print_stats(pclass_test);
   cout << endl;

   cout << "Stats for survived_test" << endl;
   print_stats(survived_test);
   cout << endl;

   cout << "Stats for sex_test" << endl;
   print_stats(sex_test);
   cout << endl;

   cout << "Stats for age_test" << endl;
   print_stats(age_test);
   cout << endl;
   

   //Naive-Bayesian

   cout << "Naive-Bayesian" << endl;


   vector<double> ap = prior(survived_train);

   cout << fixed << setprecision(6);
   cout << "Prior probability, survived = no, survived = yes." << endl;
   cout << ap[0] << " " << ap[1] << endl << endl;


   //start chrono to calculate time
   auto start = chrono::steady_clock::now();

   vector<int> count_survived(2);
   count_survived[0] = count(survived_train.begin(), survived_train.end(), 0);
   count_survived[1] = count(survived_train.begin(), survived_train.end(), 1);


   double lh_pclass[2][3];
   for (int sv = 0; sv <= 1; sv++)
      for (int pc = 1; pc <= 3; pc++)
         lh_pclass[sv][pc - 1] = (double)nrow(pclass_train, pc, survived_train, sv) / (double)count_survived[sv];

   double lh_sex[2][2];
   for (int sv = 0; sv <= 1; sv++)
      for (int sx = 0; sx <= 1; sx++)
         lh_sex[sv][sx] = (double)nrow(sex_train, sx, survived_train, sv) / (double)count_survived[sv];

   cout << "Likelihood for p(pclass|survived):" << endl;

   cout << "sex | class (1:2:3)" << endl;
   for (int i = 0; i < 2; i++)
   {
      cout << i << "    ";
      for (int j = 0; j < 3; j++)
         cout << lh_pclass[i][j] << " ";
      cout << endl;
   }

   cout << endl << "Likelihood for p(sex|survived):" << endl;
   cout << "sex | survived:not" << endl;
   for (int i = 0; i < 2; i++)
   {
      cout << i << "    ";
      for (int j = 0; j < 2; j++)
         cout << lh_sex[i][j] << " ";
      cout << endl;
   }

   vector<double> age_mean(2);
   vector<double> age_var(2);

   for (int sv = 0; sv <= 1; sv++)
   {
      age_mean[sv] = mean_condition(age_train, survived_train, sv);
      age_var[sv] = var_condition(age_train, survived_train, sv);
   }

   cout << endl;

   // end stopwatch
   auto end = chrono::steady_clock::now();



   double e = 2.7182;
   double sum;

   vector<double> raw(2);
   vector<double> probabilities(MAX_LEN);
   for (int i = 0; i < pclass_test.size(); i++) {

      raw = raw_prob(test[0][i], test[2][i], test[3][i], lh_pclass, lh_sex, ap, age_mean, age_var);
      
      sum = (pow(e, raw[0]) / (1 + pow(e, raw[0])));
      probabilities.push_back(sum);
   }

   //print accuracy, sensitivity, specificity
   double accuracy = calc_acc(survived_test, probabilities);
   double sensitivity = calc_sens(survived_test, probabilities);
   double specificity = calc_spec(survived_test, probabilities);
   cout << "Accuracy: " << accuracy << endl;
   cout << "Sensitivity: " << sensitivity << endl;
   cout << "Specificity: " << specificity << endl;

   cout << endl;
   cout << "Applied to the first 5 test observations:" << endl;

   vector<double> raw2(2);
   for (int i = 0; i < 5; i++)
   {
      raw2 = raw_prob(test[0][i], test[2][i], test[3][i], lh_pclass, lh_sex, ap, age_mean, age_var);
      cout << raw2[0] << " " << raw2[1] << endl;
   }


   // print time 
   cout << "Elapsed time in milliseconds: "
      << chrono::duration_cast<chrono::milliseconds>(end - start).count()
      << " ms" << endl;

   cout << "\nProgram terminated.";

   return 0;
}
