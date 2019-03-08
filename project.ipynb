{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Indicators of Loan Default\n",
    "## by Lucrezia Morvilli\n",
    "\n",
    "## Preliminary Wrangling\n",
    "\n",
    "> I have decided to analyse one of the datasets in the options, **Prosper Loan Data**. It includes data on 100k+ loans, with extensive information and various details on them. \n",
    "> By looking at the Data Dictionary I decided to focus on 12 features of these loans:\n",
    "> - *ListingNumber*: The number that uniquely identifies the listing to the public as displayed on the website.\n",
    "- *ListingCreationDate*: The date the listing was created.\n",
    "- *CreditGrade*: The Credit rating that was assigned at the time the listing went live. Applicable for listings pre-2009 period and will only be populated for those listings.\n",
    "- *Term*: The length of the loan expressed in months.\n",
    "- *LoanStatus*: The current status of the loan: Cancelled,  Chargedoff, Completed, Current, Defaulted, FinalPaymentInProgress, PastDue. The PastDue status will be accompanied by a delinquency bucket.\n",
    "- *ClosedDate*: Closed date is applicable for Cancelled, Completed, Chargedoff and Defaulted loan statuses.\n",
    "- *ProsperRating (numeric)*: The  Prosper Rating assigned at the time the listing was created (applicable for loans originated after July 2009): \n",
    "    -  0 - N/A\n",
    "    -  1 - HR\n",
    "    -  2 - E \n",
    "    -  3 - D\n",
    "    -  4 - C \n",
    "    -  5 - B \n",
    "    -  6 - A \n",
    "    -  7 - AA  \n",
    "- *ProsperScore*: A custom risk score built using historical Prosper data. The score ranges from 1-10, with 10 being the best, or lowest risk score.  Applicable for loans originated after July 2009.\n",
    "- *ListingCategory*: The category of the listing that the borrower selected when posting their listing: \n",
    "    -  0 - Not Available \n",
    "    -  1 - Debt Consolidation \n",
    "    -  2 - Home Improvement \n",
    "    -  3 - Business \n",
    "    -  4 - Personal Loan \n",
    "    -  5 - Student Use \n",
    "    -  6 - Auto \n",
    "    -  7 - Other \n",
    "    -  8 - Baby&Adoption \n",
    "    -  9 - Boat \n",
    "    -  10 - Cosmetic Procedure \n",
    "    -  11 - Engagement Ring \n",
    "    -  12 - Green Loans \n",
    "    -  13 - Household Expenses \n",
    "    -  14 - Large Purchases \n",
    "    -  15 - Medical/Dental \n",
    "    -  16 - Motorcycle\n",
    "    -  17 - RV \n",
    "    -  18 - Taxes \n",
    "    -  19 - Vacation\n",
    "    -  20 - Wedding Loans\n",
    "- *Occupation*: The Occupation selected by the Borrower at the time they created the listing.\n",
    "- *EmploymentStatus*: The employment status of the borrower at the time they posted the listing.\n",
    "- *IsBorrowerHomeowner*: A Borrower will be classified as a homowner if they have a mortgage on their credit profile or provide documentation confirming they are a homeowner."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import all packages and set plots to be embedded inline\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sb\n",
    "import matplotlib.dates as dates\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ListingKey</th>\n",
       "      <th>ListingNumber</th>\n",
       "      <th>ListingCreationDate</th>\n",
       "      <th>CreditGrade</th>\n",
       "      <th>Term</th>\n",
       "      <th>LoanStatus</th>\n",
       "      <th>ClosedDate</th>\n",
       "      <th>BorrowerAPR</th>\n",
       "      <th>BorrowerRate</th>\n",
       "      <th>LenderYield</th>\n",
       "      <th>...</th>\n",
       "      <th>LP_ServiceFees</th>\n",
       "      <th>LP_CollectionFees</th>\n",
       "      <th>LP_GrossPrincipalLoss</th>\n",
       "      <th>LP_NetPrincipalLoss</th>\n",
       "      <th>LP_NonPrincipalRecoverypayments</th>\n",
       "      <th>PercentFunded</th>\n",
       "      <th>Recommendations</th>\n",
       "      <th>InvestmentFromFriendsCount</th>\n",
       "      <th>InvestmentFromFriendsAmount</th>\n",
       "      <th>Investors</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1021339766868145413AB3B</td>\n",
       "      <td>193129</td>\n",
       "      <td>2007-08-26 19:09:29.263000000</td>\n",
       "      <td>C</td>\n",
       "      <td>36</td>\n",
       "      <td>Completed</td>\n",
       "      <td>2009-08-14 00:00:00</td>\n",
       "      <td>0.16516</td>\n",
       "      <td>0.1580</td>\n",
       "      <td>0.1380</td>\n",
       "      <td>...</td>\n",
       "      <td>-133.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10273602499503308B223C1</td>\n",
       "      <td>1209647</td>\n",
       "      <td>2014-02-27 08:28:07.900000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.12016</td>\n",
       "      <td>0.0920</td>\n",
       "      <td>0.0820</td>\n",
       "      <td>...</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0EE9337825851032864889A</td>\n",
       "      <td>81716</td>\n",
       "      <td>2007-01-05 15:00:47.090000000</td>\n",
       "      <td>HR</td>\n",
       "      <td>36</td>\n",
       "      <td>Completed</td>\n",
       "      <td>2009-12-17 00:00:00</td>\n",
       "      <td>0.28269</td>\n",
       "      <td>0.2750</td>\n",
       "      <td>0.2400</td>\n",
       "      <td>...</td>\n",
       "      <td>-24.20</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0EF5356002482715299901A</td>\n",
       "      <td>658116</td>\n",
       "      <td>2012-10-22 11:02:35.010000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.12528</td>\n",
       "      <td>0.0974</td>\n",
       "      <td>0.0874</td>\n",
       "      <td>...</td>\n",
       "      <td>-108.01</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>158</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0F023589499656230C5E3E2</td>\n",
       "      <td>909464</td>\n",
       "      <td>2013-09-14 18:38:39.097000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.24614</td>\n",
       "      <td>0.2085</td>\n",
       "      <td>0.1985</td>\n",
       "      <td>...</td>\n",
       "      <td>-60.27</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 81 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                ListingKey  ListingNumber            ListingCreationDate  \\\n",
       "0  1021339766868145413AB3B         193129  2007-08-26 19:09:29.263000000   \n",
       "1  10273602499503308B223C1        1209647  2014-02-27 08:28:07.900000000   \n",
       "2  0EE9337825851032864889A          81716  2007-01-05 15:00:47.090000000   \n",
       "3  0EF5356002482715299901A         658116  2012-10-22 11:02:35.010000000   \n",
       "4  0F023589499656230C5E3E2         909464  2013-09-14 18:38:39.097000000   \n",
       "\n",
       "  CreditGrade  Term LoanStatus           ClosedDate  BorrowerAPR  \\\n",
       "0           C    36  Completed  2009-08-14 00:00:00      0.16516   \n",
       "1         NaN    36    Current                  NaN      0.12016   \n",
       "2          HR    36  Completed  2009-12-17 00:00:00      0.28269   \n",
       "3         NaN    36    Current                  NaN      0.12528   \n",
       "4         NaN    36    Current                  NaN      0.24614   \n",
       "\n",
       "   BorrowerRate  LenderYield    ...     LP_ServiceFees  LP_CollectionFees  \\\n",
       "0        0.1580       0.1380    ...            -133.18                0.0   \n",
       "1        0.0920       0.0820    ...               0.00                0.0   \n",
       "2        0.2750       0.2400    ...             -24.20                0.0   \n",
       "3        0.0974       0.0874    ...            -108.01                0.0   \n",
       "4        0.2085       0.1985    ...             -60.27                0.0   \n",
       "\n",
       "   LP_GrossPrincipalLoss  LP_NetPrincipalLoss LP_NonPrincipalRecoverypayments  \\\n",
       "0                    0.0                  0.0                             0.0   \n",
       "1                    0.0                  0.0                             0.0   \n",
       "2                    0.0                  0.0                             0.0   \n",
       "3                    0.0                  0.0                             0.0   \n",
       "4                    0.0                  0.0                             0.0   \n",
       "\n",
       "   PercentFunded  Recommendations InvestmentFromFriendsCount  \\\n",
       "0            1.0                0                          0   \n",
       "1            1.0                0                          0   \n",
       "2            1.0                0                          0   \n",
       "3            1.0                0                          0   \n",
       "4            1.0                0                          0   \n",
       "\n",
       "  InvestmentFromFriendsAmount Investors  \n",
       "0                         0.0       258  \n",
       "1                         0.0         1  \n",
       "2                         0.0        41  \n",
       "3                         0.0       158  \n",
       "4                         0.0        20  \n",
       "\n",
       "[5 rows x 81 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# load dataset\n",
    "df = pd.read_csv('prosperLoanData.csv')\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Below I will perform some initial exploratory analysis of the dataset, in order to understand more regarding the type of data it contains."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(113937, 81)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 113937 entries, 0 to 113936\n",
      "Data columns (total 81 columns):\n",
      "ListingKey                             113937 non-null object\n",
      "ListingNumber                          113937 non-null int64\n",
      "ListingCreationDate                    113937 non-null object\n",
      "CreditGrade                            28953 non-null object\n",
      "Term                                   113937 non-null int64\n",
      "LoanStatus                             113937 non-null object\n",
      "ClosedDate                             55089 non-null object\n",
      "BorrowerAPR                            113912 non-null float64\n",
      "BorrowerRate                           113937 non-null float64\n",
      "LenderYield                            113937 non-null float64\n",
      "EstimatedEffectiveYield                84853 non-null float64\n",
      "EstimatedLoss                          84853 non-null float64\n",
      "EstimatedReturn                        84853 non-null float64\n",
      "ProsperRating (numeric)                84853 non-null float64\n",
      "ProsperRating (Alpha)                  84853 non-null object\n",
      "ProsperScore                           84853 non-null float64\n",
      "ListingCategory (numeric)              113937 non-null int64\n",
      "BorrowerState                          108422 non-null object\n",
      "Occupation                             110349 non-null object\n",
      "EmploymentStatus                       111682 non-null object\n",
      "EmploymentStatusDuration               106312 non-null float64\n",
      "IsBorrowerHomeowner                    113937 non-null bool\n",
      "CurrentlyInGroup                       113937 non-null bool\n",
      "GroupKey                               13341 non-null object\n",
      "DateCreditPulled                       113937 non-null object\n",
      "CreditScoreRangeLower                  113346 non-null float64\n",
      "CreditScoreRangeUpper                  113346 non-null float64\n",
      "FirstRecordedCreditLine                113240 non-null object\n",
      "CurrentCreditLines                     106333 non-null float64\n",
      "OpenCreditLines                        106333 non-null float64\n",
      "TotalCreditLinespast7years             113240 non-null float64\n",
      "OpenRevolvingAccounts                  113937 non-null int64\n",
      "OpenRevolvingMonthlyPayment            113937 non-null float64\n",
      "InquiriesLast6Months                   113240 non-null float64\n",
      "TotalInquiries                         112778 non-null float64\n",
      "CurrentDelinquencies                   113240 non-null float64\n",
      "AmountDelinquent                       106315 non-null float64\n",
      "DelinquenciesLast7Years                112947 non-null float64\n",
      "PublicRecordsLast10Years               113240 non-null float64\n",
      "PublicRecordsLast12Months              106333 non-null float64\n",
      "RevolvingCreditBalance                 106333 non-null float64\n",
      "BankcardUtilization                    106333 non-null float64\n",
      "AvailableBankcardCredit                106393 non-null float64\n",
      "TotalTrades                            106393 non-null float64\n",
      "TradesNeverDelinquent (percentage)     106393 non-null float64\n",
      "TradesOpenedLast6Months                106393 non-null float64\n",
      "DebtToIncomeRatio                      105383 non-null float64\n",
      "IncomeRange                            113937 non-null object\n",
      "IncomeVerifiable                       113937 non-null bool\n",
      "StatedMonthlyIncome                    113937 non-null float64\n",
      "LoanKey                                113937 non-null object\n",
      "TotalProsperLoans                      22085 non-null float64\n",
      "TotalProsperPaymentsBilled             22085 non-null float64\n",
      "OnTimeProsperPayments                  22085 non-null float64\n",
      "ProsperPaymentsLessThanOneMonthLate    22085 non-null float64\n",
      "ProsperPaymentsOneMonthPlusLate        22085 non-null float64\n",
      "ProsperPrincipalBorrowed               22085 non-null float64\n",
      "ProsperPrincipalOutstanding            22085 non-null float64\n",
      "ScorexChangeAtTimeOfListing            18928 non-null float64\n",
      "LoanCurrentDaysDelinquent              113937 non-null int64\n",
      "LoanFirstDefaultedCycleNumber          16952 non-null float64\n",
      "LoanMonthsSinceOrigination             113937 non-null int64\n",
      "LoanNumber                             113937 non-null int64\n",
      "LoanOriginalAmount                     113937 non-null int64\n",
      "LoanOriginationDate                    113937 non-null object\n",
      "LoanOriginationQuarter                 113937 non-null object\n",
      "MemberKey                              113937 non-null object\n",
      "MonthlyLoanPayment                     113937 non-null float64\n",
      "LP_CustomerPayments                    113937 non-null float64\n",
      "LP_CustomerPrincipalPayments           113937 non-null float64\n",
      "LP_InterestandFees                     113937 non-null float64\n",
      "LP_ServiceFees                         113937 non-null float64\n",
      "LP_CollectionFees                      113937 non-null float64\n",
      "LP_GrossPrincipalLoss                  113937 non-null float64\n",
      "LP_NetPrincipalLoss                    113937 non-null float64\n",
      "LP_NonPrincipalRecoverypayments        113937 non-null float64\n",
      "PercentFunded                          113937 non-null float64\n",
      "Recommendations                        113937 non-null int64\n",
      "InvestmentFromFriendsCount             113937 non-null int64\n",
      "InvestmentFromFriendsAmount            113937 non-null float64\n",
      "Investors                              113937 non-null int64\n",
      "dtypes: bool(3), float64(50), int64(11), object(17)\n",
      "memory usage: 68.1+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cleaning ##\n",
    "Here I will define the data I will analyse and perform some quick cleaning actions before looking at loan features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# define df that will be analysed\n",
    "df_loans = df[['ListingNumber', 'ListingCreationDate', 'CreditGrade', 'Term', 'LoanStatus', 'ClosedDate', 'ProsperRating (numeric)', 'ProsperScore', 'ListingCategory (numeric)', 'Occupation', 'EmploymentStatus', 'IsBorrowerHomeowner']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(113937, 12)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Other                                 28617\n",
       "Professional                          13628\n",
       "Computer Programmer                    4478\n",
       "Executive                              4311\n",
       "Teacher                                3759\n",
       "Administrative Assistant               3688\n",
       "Analyst                                3602\n",
       "Sales - Commission                     3446\n",
       "Accountant/CPA                         3233\n",
       "Clerical                               3164\n",
       "Sales - Retail                         2797\n",
       "Skilled Labor                          2746\n",
       "Retail Management                      2602\n",
       "Nurse (RN)                             2489\n",
       "Construction                           1790\n",
       "Truck Driver                           1675\n",
       "Laborer                                1595\n",
       "Police Officer/Correction Officer      1578\n",
       "Civil Service                          1457\n",
       "Engineer - Mechanical                  1406\n",
       "Military Enlisted                      1272\n",
       "Food Service Management                1239\n",
       "Engineer - Electrical                  1125\n",
       "Food Service                           1123\n",
       "Medical Technician                     1117\n",
       "Attorney                               1046\n",
       "Tradesman - Mechanic                    951\n",
       "Social Worker                           741\n",
       "Postal Service                          627\n",
       "Professor                               557\n",
       "                                      ...  \n",
       "Scientist                               372\n",
       "Military Officer                        346\n",
       "Bus Driver                              316\n",
       "Principal                               312\n",
       "Teacher's Aide                          276\n",
       "Pharmacist                              257\n",
       "Student - College Graduate Student      245\n",
       "Landscaping                             236\n",
       "Engineer - Chemical                     225\n",
       "Investor                                214\n",
       "Architect                               213\n",
       "Pilot - Private/Commercial              199\n",
       "Clergy                                  196\n",
       "Student - College Senior                188\n",
       "Car Dealer                              180\n",
       "Chemist                                 145\n",
       "Psychologist                            145\n",
       "Biologist                               125\n",
       "Religious                               124\n",
       "Flight Attendant                        123\n",
       "Homemaker                               120\n",
       "Tradesman - Carpenter                   120\n",
       "Student - College Junior                112\n",
       "Tradesman - Plumber                     102\n",
       "Student - College Sophomore              69\n",
       "Dentist                                  68\n",
       "Student - College Freshman               41\n",
       "Student - Community College              28\n",
       "Judge                                    22\n",
       "Student - Technical School               16\n",
       "Name: Occupation, Length: 67, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.Occupation.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "67"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.Occupation.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Employed         67322\n",
       "Full-time        26355\n",
       "Self-employed     6134\n",
       "Not available     5347\n",
       "Other             3806\n",
       "Part-time         1088\n",
       "Not employed       835\n",
       "Retired            795\n",
       "Name: EmploymentStatus, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.EmploymentStatus.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "C     5649\n",
       "D     5153\n",
       "B     4389\n",
       "AA    3509\n",
       "HR    3508\n",
       "A     3315\n",
       "E     3289\n",
       "NC     141\n",
       "Name: CreditGrade, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.CreditGrade.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "871"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans['ListingNumber'].duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ListingNumber</th>\n",
       "      <th>ListingCreationDate</th>\n",
       "      <th>CreditGrade</th>\n",
       "      <th>Term</th>\n",
       "      <th>LoanStatus</th>\n",
       "      <th>ClosedDate</th>\n",
       "      <th>ProsperRating (numeric)</th>\n",
       "      <th>ProsperScore</th>\n",
       "      <th>ListingCategory (numeric)</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>EmploymentStatus</th>\n",
       "      <th>IsBorrowerHomeowner</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1023355</td>\n",
       "      <td>2013-12-02 10:43:39.117000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>7</td>\n",
       "      <td>Food Service</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1023355</td>\n",
       "      <td>2013-12-02 10:43:39.117000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>7</td>\n",
       "      <td>Food Service</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ListingNumber            ListingCreationDate CreditGrade  Term LoanStatus  \\\n",
       "8        1023355  2013-12-02 10:43:39.117000000         NaN    36    Current   \n",
       "9        1023355  2013-12-02 10:43:39.117000000         NaN    36    Current   \n",
       "\n",
       "  ClosedDate  ProsperRating (numeric)  ProsperScore  \\\n",
       "8        NaN                      7.0           9.0   \n",
       "9        NaN                      7.0          11.0   \n",
       "\n",
       "   ListingCategory (numeric)    Occupation EmploymentStatus  \\\n",
       "8                          7  Food Service         Employed   \n",
       "9                          7  Food Service         Employed   \n",
       "\n",
       "   IsBorrowerHomeowner  \n",
       "8                 True  \n",
       "9                 True  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking one duplicate listing\n",
    "df_loans[df_loans['ListingNumber']==1023355]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ListingNumber</th>\n",
       "      <th>ListingCreationDate</th>\n",
       "      <th>CreditGrade</th>\n",
       "      <th>Term</th>\n",
       "      <th>LoanStatus</th>\n",
       "      <th>ClosedDate</th>\n",
       "      <th>ProsperRating (numeric)</th>\n",
       "      <th>ProsperScore</th>\n",
       "      <th>ListingCategory (numeric)</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>EmploymentStatus</th>\n",
       "      <th>IsBorrowerHomeowner</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1023355</td>\n",
       "      <td>2013-12-02 10:43:39.117000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>7</td>\n",
       "      <td>Food Service</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>69</th>\n",
       "      <td>1162592</td>\n",
       "      <td>2014-01-25 12:07:54.537000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Investor</td>\n",
       "      <td>Self-employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>158</th>\n",
       "      <td>1202850</td>\n",
       "      <td>2014-02-12 16:31:25.340000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>6</td>\n",
       "      <td>Tradesman - Mechanic</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>224</th>\n",
       "      <td>1130508</td>\n",
       "      <td>2014-01-10 07:24:44.853000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Sales - Commission</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>236</th>\n",
       "      <td>973657</td>\n",
       "      <td>2013-11-02 01:29:09.810000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Tradesman - Mechanic</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>277</th>\n",
       "      <td>1220331</td>\n",
       "      <td>2014-02-18 12:59:08.680000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Full-time</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>476</th>\n",
       "      <td>1244304</td>\n",
       "      <td>2014-03-06 08:54:42.840000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>491</th>\n",
       "      <td>1116282</td>\n",
       "      <td>2014-01-06 07:18:54.173000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Analyst</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>532</th>\n",
       "      <td>1150992</td>\n",
       "      <td>2014-01-20 10:51:29.120000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Other</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>554</th>\n",
       "      <td>1054630</td>\n",
       "      <td>2013-12-17 22:15:59.470000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>7</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>633</th>\n",
       "      <td>1079885</td>\n",
       "      <td>2013-12-15 12:25:42.193000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Analyst</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>690</th>\n",
       "      <td>991235</td>\n",
       "      <td>2013-11-03 08:48:26.140000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>765</th>\n",
       "      <td>1022836</td>\n",
       "      <td>2013-12-02 07:36:43.197000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Computer Programmer</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>771</th>\n",
       "      <td>993658</td>\n",
       "      <td>2013-11-13 19:46:34.570000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Nurse (RN)</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>854</th>\n",
       "      <td>1082097</td>\n",
       "      <td>2013-12-12 11:08:56.630000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>1137096</td>\n",
       "      <td>2014-01-14 09:02:45.567000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>901</th>\n",
       "      <td>1026799</td>\n",
       "      <td>2013-12-03 00:36:12.727000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1017</th>\n",
       "      <td>1236380</td>\n",
       "      <td>2014-03-06 03:45:19.167000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Other</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1085</th>\n",
       "      <td>1079667</td>\n",
       "      <td>2013-12-12 03:37:30.163000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Sales - Retail</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1153</th>\n",
       "      <td>1086246</td>\n",
       "      <td>2013-12-13 01:35:00.870000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1460</th>\n",
       "      <td>998145</td>\n",
       "      <td>2013-10-23 07:34:42.750000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Analyst</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1842</th>\n",
       "      <td>1192663</td>\n",
       "      <td>2014-03-03 10:18:08.083000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>13</td>\n",
       "      <td>Analyst</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1892</th>\n",
       "      <td>1054235</td>\n",
       "      <td>2013-12-05 12:28:21.723000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Computer Programmer</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1962</th>\n",
       "      <td>911082</td>\n",
       "      <td>2013-09-19 11:47:58.917000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1972</th>\n",
       "      <td>968697</td>\n",
       "      <td>2013-10-09 15:26:16.583000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>18</td>\n",
       "      <td>Accountant/CPA</td>\n",
       "      <td>Self-employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2048</th>\n",
       "      <td>1052408</td>\n",
       "      <td>2013-12-05 09:18:30.973000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2097</th>\n",
       "      <td>1069756</td>\n",
       "      <td>2013-12-31 10:34:15.973000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2181</th>\n",
       "      <td>1107074</td>\n",
       "      <td>2013-12-30 16:58:06.427000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Scientist</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2288</th>\n",
       "      <td>1166956</td>\n",
       "      <td>2014-02-19 14:11:22.497000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2487</th>\n",
       "      <td>1026919</td>\n",
       "      <td>2013-12-05 04:50:07.360000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Computer Programmer</td>\n",
       "      <td>Self-employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111771</th>\n",
       "      <td>1125892</td>\n",
       "      <td>2014-01-21 14:48:16.600000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111890</th>\n",
       "      <td>827053</td>\n",
       "      <td>2013-07-02 14:09:36.273000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Construction</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111938</th>\n",
       "      <td>1005026</td>\n",
       "      <td>2013-11-09 05:52:10.357000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>6</td>\n",
       "      <td>Engineer - Electrical</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112024</th>\n",
       "      <td>1014469</td>\n",
       "      <td>2013-11-22 18:22:02.563000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>20</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112098</th>\n",
       "      <td>1086346</td>\n",
       "      <td>2014-01-06 06:25:23.533000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112124</th>\n",
       "      <td>912223</td>\n",
       "      <td>2013-09-19 12:44:10.020000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Accountant/CPA</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112316</th>\n",
       "      <td>1056467</td>\n",
       "      <td>2013-12-05 22:05:04.813000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Police Officer/Correction Officer</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112549</th>\n",
       "      <td>1018934</td>\n",
       "      <td>2013-11-14 19:24:22.833000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112646</th>\n",
       "      <td>979285</td>\n",
       "      <td>2013-11-04 14:12:13.310000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>13</td>\n",
       "      <td>Doctor</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112653</th>\n",
       "      <td>1146712</td>\n",
       "      <td>2014-02-07 21:40:58.917000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>7</td>\n",
       "      <td>Civil Service</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112684</th>\n",
       "      <td>1027866</td>\n",
       "      <td>2013-11-08 05:04:32.807000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Retail Management</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112774</th>\n",
       "      <td>1091854</td>\n",
       "      <td>2014-01-07 14:18:59.403000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Executive</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112775</th>\n",
       "      <td>1169617</td>\n",
       "      <td>2014-02-19 20:32:03.660000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112808</th>\n",
       "      <td>1040752</td>\n",
       "      <td>2013-12-09 13:56:51.750000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Pilot - Private/Commercial</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112857</th>\n",
       "      <td>971240</td>\n",
       "      <td>2013-10-28 02:32:23.193000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Computer Programmer</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112909</th>\n",
       "      <td>1032395</td>\n",
       "      <td>2013-11-25 08:06:35.157000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Sales - Retail</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112921</th>\n",
       "      <td>970853</td>\n",
       "      <td>2013-10-24 19:39:25.217000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Executive</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112990</th>\n",
       "      <td>950327</td>\n",
       "      <td>2013-10-11 07:26:59.320000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113044</th>\n",
       "      <td>1125301</td>\n",
       "      <td>2014-01-21 13:25:51.740000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113167</th>\n",
       "      <td>1197569</td>\n",
       "      <td>2014-02-15 20:36:09.020000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>18</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113248</th>\n",
       "      <td>1026285</td>\n",
       "      <td>2013-11-06 16:49:49.477000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Sales - Commission</td>\n",
       "      <td>Self-employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113406</th>\n",
       "      <td>1027883</td>\n",
       "      <td>2013-11-21 20:52:36.510000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>19</td>\n",
       "      <td>Attorney</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113570</th>\n",
       "      <td>1076210</td>\n",
       "      <td>2013-12-14 12:55:45.637000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Attorney</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113706</th>\n",
       "      <td>1018247</td>\n",
       "      <td>2013-11-14 16:21:07.883000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Pharmacist</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113750</th>\n",
       "      <td>1190813</td>\n",
       "      <td>2014-02-13 09:02:27.677000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Civil Service</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113762</th>\n",
       "      <td>1040844</td>\n",
       "      <td>2013-11-18 17:55:25.090000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>20</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113784</th>\n",
       "      <td>1021056</td>\n",
       "      <td>2013-11-05 20:51:45.540000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113855</th>\n",
       "      <td>1211163</td>\n",
       "      <td>2014-02-15 18:43:57.650000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Food Service Management</td>\n",
       "      <td>Full-time</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113881</th>\n",
       "      <td>967344</td>\n",
       "      <td>2013-10-09 10:09:28.530000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>7</td>\n",
       "      <td>Professional</td>\n",
       "      <td>Employed</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113912</th>\n",
       "      <td>1083677</td>\n",
       "      <td>2013-12-16 16:36:00.990000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>36</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Scientist</td>\n",
       "      <td>Employed</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1456 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        ListingNumber            ListingCreationDate CreditGrade  Term  \\\n",
       "9             1023355  2013-12-02 10:43:39.117000000         NaN    36   \n",
       "69            1162592  2014-01-25 12:07:54.537000000         NaN    60   \n",
       "158           1202850  2014-02-12 16:31:25.340000000         NaN    36   \n",
       "224           1130508  2014-01-10 07:24:44.853000000         NaN    36   \n",
       "236            973657  2013-11-02 01:29:09.810000000         NaN    36   \n",
       "277           1220331  2014-02-18 12:59:08.680000000         NaN    60   \n",
       "476           1244304  2014-03-06 08:54:42.840000000         NaN    60   \n",
       "491           1116282  2014-01-06 07:18:54.173000000         NaN    60   \n",
       "532           1150992  2014-01-20 10:51:29.120000000         NaN    36   \n",
       "554           1054630  2013-12-17 22:15:59.470000000         NaN    60   \n",
       "633           1079885  2013-12-15 12:25:42.193000000         NaN    36   \n",
       "690            991235  2013-11-03 08:48:26.140000000         NaN    60   \n",
       "765           1022836  2013-12-02 07:36:43.197000000         NaN    60   \n",
       "771            993658  2013-11-13 19:46:34.570000000         NaN    36   \n",
       "854           1082097  2013-12-12 11:08:56.630000000         NaN    60   \n",
       "890           1137096  2014-01-14 09:02:45.567000000         NaN    60   \n",
       "901           1026799  2013-12-03 00:36:12.727000000         NaN    60   \n",
       "1017          1236380  2014-03-06 03:45:19.167000000         NaN    36   \n",
       "1085          1079667  2013-12-12 03:37:30.163000000         NaN    36   \n",
       "1153          1086246  2013-12-13 01:35:00.870000000         NaN    36   \n",
       "1460           998145  2013-10-23 07:34:42.750000000         NaN    36   \n",
       "1842          1192663  2014-03-03 10:18:08.083000000         NaN    36   \n",
       "1892          1054235  2013-12-05 12:28:21.723000000         NaN    36   \n",
       "1962           911082  2013-09-19 11:47:58.917000000         NaN    60   \n",
       "1972           968697  2013-10-09 15:26:16.583000000         NaN    36   \n",
       "2048          1052408  2013-12-05 09:18:30.973000000         NaN    36   \n",
       "2097          1069756  2013-12-31 10:34:15.973000000         NaN    60   \n",
       "2181          1107074  2013-12-30 16:58:06.427000000         NaN    60   \n",
       "2288          1166956  2014-02-19 14:11:22.497000000         NaN    36   \n",
       "2487          1026919  2013-12-05 04:50:07.360000000         NaN    36   \n",
       "...               ...                            ...         ...   ...   \n",
       "111771        1125892  2014-01-21 14:48:16.600000000         NaN    36   \n",
       "111890         827053  2013-07-02 14:09:36.273000000         NaN    60   \n",
       "111938        1005026  2013-11-09 05:52:10.357000000         NaN    60   \n",
       "112024        1014469  2013-11-22 18:22:02.563000000         NaN    36   \n",
       "112098        1086346  2014-01-06 06:25:23.533000000         NaN    36   \n",
       "112124         912223  2013-09-19 12:44:10.020000000         NaN    36   \n",
       "112316        1056467  2013-12-05 22:05:04.813000000         NaN    36   \n",
       "112549        1018934  2013-11-14 19:24:22.833000000         NaN    60   \n",
       "112646         979285  2013-11-04 14:12:13.310000000         NaN    36   \n",
       "112653        1146712  2014-02-07 21:40:58.917000000         NaN    36   \n",
       "112684        1027866  2013-11-08 05:04:32.807000000         NaN    60   \n",
       "112774        1091854  2014-01-07 14:18:59.403000000         NaN    36   \n",
       "112775        1169617  2014-02-19 20:32:03.660000000         NaN    60   \n",
       "112808        1040752  2013-12-09 13:56:51.750000000         NaN    36   \n",
       "112857         971240  2013-10-28 02:32:23.193000000         NaN    60   \n",
       "112909        1032395  2013-11-25 08:06:35.157000000         NaN    60   \n",
       "112921         970853  2013-10-24 19:39:25.217000000         NaN    36   \n",
       "112990         950327  2013-10-11 07:26:59.320000000         NaN    60   \n",
       "113044        1125301  2014-01-21 13:25:51.740000000         NaN    60   \n",
       "113167        1197569  2014-02-15 20:36:09.020000000         NaN    36   \n",
       "113248        1026285  2013-11-06 16:49:49.477000000         NaN    60   \n",
       "113406        1027883  2013-11-21 20:52:36.510000000         NaN    36   \n",
       "113570        1076210  2013-12-14 12:55:45.637000000         NaN    36   \n",
       "113706        1018247  2013-11-14 16:21:07.883000000         NaN    36   \n",
       "113750        1190813  2014-02-13 09:02:27.677000000         NaN    36   \n",
       "113762        1040844  2013-11-18 17:55:25.090000000         NaN    60   \n",
       "113784        1021056  2013-11-05 20:51:45.540000000         NaN    36   \n",
       "113855        1211163  2014-02-15 18:43:57.650000000         NaN    36   \n",
       "113881         967344  2013-10-09 10:09:28.530000000         NaN    60   \n",
       "113912        1083677  2013-12-16 16:36:00.990000000         NaN    36   \n",
       "\n",
       "       LoanStatus ClosedDate  ProsperRating (numeric)  ProsperScore  \\\n",
       "9         Current        NaN                      7.0          11.0   \n",
       "69        Current        NaN                      6.0          11.0   \n",
       "158       Current        NaN                      7.0          11.0   \n",
       "224       Current        NaN                      6.0          11.0   \n",
       "236       Current        NaN                      7.0          11.0   \n",
       "277       Current        NaN                      5.0          11.0   \n",
       "476       Current        NaN                      5.0          11.0   \n",
       "491       Current        NaN                      5.0          11.0   \n",
       "532       Current        NaN                      6.0          11.0   \n",
       "554       Current        NaN                      7.0          11.0   \n",
       "633       Current        NaN                      6.0          11.0   \n",
       "690       Current        NaN                      6.0          11.0   \n",
       "765       Current        NaN                      5.0          11.0   \n",
       "771       Current        NaN                      7.0          11.0   \n",
       "854       Current        NaN                      6.0          11.0   \n",
       "890       Current        NaN                      5.0          11.0   \n",
       "901       Current        NaN                      6.0          11.0   \n",
       "1017      Current        NaN                      7.0          11.0   \n",
       "1085      Current        NaN                      7.0          11.0   \n",
       "1153      Current        NaN                      7.0          11.0   \n",
       "1460      Current        NaN                      6.0          11.0   \n",
       "1842      Current        NaN                      7.0          11.0   \n",
       "1892      Current        NaN                      6.0          11.0   \n",
       "1962      Current        NaN                      6.0          11.0   \n",
       "1972      Current        NaN                      5.0          11.0   \n",
       "2048      Current        NaN                      7.0          11.0   \n",
       "2097      Current        NaN                      6.0          11.0   \n",
       "2181      Current        NaN                      4.0          11.0   \n",
       "2288      Current        NaN                      6.0          11.0   \n",
       "2487      Current        NaN                      7.0          11.0   \n",
       "...           ...        ...                      ...           ...   \n",
       "111771    Current        NaN                      7.0          11.0   \n",
       "111890    Current        NaN                      5.0          11.0   \n",
       "111938    Current        NaN                      5.0          11.0   \n",
       "112024    Current        NaN                      7.0          11.0   \n",
       "112098    Current        NaN                      6.0          11.0   \n",
       "112124    Current        NaN                      7.0          11.0   \n",
       "112316    Current        NaN                      7.0          11.0   \n",
       "112549    Current        NaN                      6.0          11.0   \n",
       "112646    Current        NaN                      7.0          11.0   \n",
       "112653    Current        NaN                      7.0          11.0   \n",
       "112684    Current        NaN                      5.0          11.0   \n",
       "112774    Current        NaN                      7.0          11.0   \n",
       "112775    Current        NaN                      7.0          11.0   \n",
       "112808    Current        NaN                      7.0          11.0   \n",
       "112857    Current        NaN                      6.0          11.0   \n",
       "112909    Current        NaN                      6.0          11.0   \n",
       "112921    Current        NaN                      7.0          11.0   \n",
       "112990    Current        NaN                      6.0          11.0   \n",
       "113044    Current        NaN                      6.0          11.0   \n",
       "113167    Current        NaN                      6.0          11.0   \n",
       "113248    Current        NaN                      6.0          11.0   \n",
       "113406    Current        NaN                      7.0          11.0   \n",
       "113570    Current        NaN                      7.0          11.0   \n",
       "113706    Current        NaN                      6.0          11.0   \n",
       "113750    Current        NaN                      7.0          11.0   \n",
       "113762    Current        NaN                      6.0          11.0   \n",
       "113784    Current        NaN                      6.0          11.0   \n",
       "113855    Current        NaN                      6.0          11.0   \n",
       "113881    Current        NaN                      4.0          11.0   \n",
       "113912    Current        NaN                      7.0          11.0   \n",
       "\n",
       "        ListingCategory (numeric)                         Occupation  \\\n",
       "9                               7                       Food Service   \n",
       "69                              1                           Investor   \n",
       "158                             6               Tradesman - Mechanic   \n",
       "224                             1                 Sales - Commission   \n",
       "236                             1               Tradesman - Mechanic   \n",
       "277                             1                              Other   \n",
       "476                             1                              Other   \n",
       "491                             1                            Analyst   \n",
       "532                             1                                NaN   \n",
       "554                             7                              Other   \n",
       "633                             1                            Analyst   \n",
       "690                             2                              Other   \n",
       "765                             1                Computer Programmer   \n",
       "771                             1                         Nurse (RN)   \n",
       "854                             1                       Professional   \n",
       "890                             3                              Other   \n",
       "901                             2                       Professional   \n",
       "1017                            1                                NaN   \n",
       "1085                            1                     Sales - Retail   \n",
       "1153                            1                              Other   \n",
       "1460                            1                            Analyst   \n",
       "1842                           13                            Analyst   \n",
       "1892                            2                Computer Programmer   \n",
       "1962                            3                       Professional   \n",
       "1972                           18                     Accountant/CPA   \n",
       "2048                            1                       Professional   \n",
       "2097                            1                       Professional   \n",
       "2181                            1                          Scientist   \n",
       "2288                            1                       Professional   \n",
       "2487                            3                Computer Programmer   \n",
       "...                           ...                                ...   \n",
       "111771                          1                       Professional   \n",
       "111890                          1                       Construction   \n",
       "111938                          6              Engineer - Electrical   \n",
       "112024                         20                       Professional   \n",
       "112098                          1                              Other   \n",
       "112124                          1                     Accountant/CPA   \n",
       "112316                          1  Police Officer/Correction Officer   \n",
       "112549                          1                              Other   \n",
       "112646                         13                             Doctor   \n",
       "112653                          7                      Civil Service   \n",
       "112684                          1                  Retail Management   \n",
       "112774                          1                          Executive   \n",
       "112775                          1                              Other   \n",
       "112808                          1         Pilot - Private/Commercial   \n",
       "112857                          1                Computer Programmer   \n",
       "112909                          2                     Sales - Retail   \n",
       "112921                          1                          Executive   \n",
       "112990                          1                       Professional   \n",
       "113044                          1                       Professional   \n",
       "113167                         18                       Professional   \n",
       "113248                          2                 Sales - Commission   \n",
       "113406                         19                           Attorney   \n",
       "113570                          1                           Attorney   \n",
       "113706                          1                         Pharmacist   \n",
       "113750                          2                      Civil Service   \n",
       "113762                         20                              Other   \n",
       "113784                          1                              Other   \n",
       "113855                          3            Food Service Management   \n",
       "113881                          7                       Professional   \n",
       "113912                          2                          Scientist   \n",
       "\n",
       "       EmploymentStatus  IsBorrowerHomeowner  \n",
       "9              Employed                 True  \n",
       "69        Self-employed                 True  \n",
       "158            Employed                False  \n",
       "224            Employed                 True  \n",
       "236            Employed                 True  \n",
       "277           Full-time                False  \n",
       "476            Employed                False  \n",
       "491            Employed                 True  \n",
       "532               Other                 True  \n",
       "554            Employed                 True  \n",
       "633            Employed                 True  \n",
       "690            Employed                 True  \n",
       "765            Employed                 True  \n",
       "771            Employed                 True  \n",
       "854            Employed                False  \n",
       "890            Employed                 True  \n",
       "901            Employed                 True  \n",
       "1017              Other                 True  \n",
       "1085           Employed                 True  \n",
       "1153           Employed                 True  \n",
       "1460           Employed                False  \n",
       "1842           Employed                 True  \n",
       "1892           Employed                 True  \n",
       "1962           Employed                 True  \n",
       "1972      Self-employed                 True  \n",
       "2048           Employed                 True  \n",
       "2097           Employed                 True  \n",
       "2181           Employed                 True  \n",
       "2288           Employed                 True  \n",
       "2487      Self-employed                 True  \n",
       "...                 ...                  ...  \n",
       "111771         Employed                 True  \n",
       "111890         Employed                 True  \n",
       "111938         Employed                False  \n",
       "112024         Employed                 True  \n",
       "112098         Employed                False  \n",
       "112124         Employed                 True  \n",
       "112316         Employed                 True  \n",
       "112549         Employed                 True  \n",
       "112646         Employed                False  \n",
       "112653         Employed                 True  \n",
       "112684         Employed                 True  \n",
       "112774         Employed                 True  \n",
       "112775         Employed                False  \n",
       "112808         Employed                 True  \n",
       "112857         Employed                False  \n",
       "112909         Employed                False  \n",
       "112921         Employed                 True  \n",
       "112990         Employed                False  \n",
       "113044         Employed                 True  \n",
       "113167         Employed                 True  \n",
       "113248    Self-employed                False  \n",
       "113406         Employed                False  \n",
       "113570         Employed                 True  \n",
       "113706         Employed                False  \n",
       "113750         Employed                 True  \n",
       "113762         Employed                 True  \n",
       "113784         Employed                 True  \n",
       "113855        Full-time                False  \n",
       "113881         Employed                 True  \n",
       "113912         Employed                False  \n",
       "\n",
       "[1456 rows x 12 columns]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans[df_loans['ProsperScore']>10]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> From the Data Dictionary we know that the *Prosper Score* should range from 1 to 10. We should check that this is true in the data and delete the lines with *Prosper Score* above 10 (as we have enough data anyway). We will also check that *Prosper Rating* and *Listing Category* have valid values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "list_drop = df_loans[df_loans['ProsperScore']>10]['ListingNumber'].index.tolist()\n",
    "\n",
    "for t in list_drop:\n",
    "    df_loans= df_loans.drop(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(112481, 12)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans = df_loans.reset_index(drop = True)\n",
    "df_loans.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#remove the listings for which we have neither Credit Grade nor Prosper Rating\n",
    "list_drop_credit = df_loans[(df_loans['CreditGrade'].isnull())&(df_loans['ProsperRating (numeric)'].isnull())].index.tolist()\n",
    "for t in list_drop_credit:\n",
    "    df_loans= df_loans.drop(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(112350, 12)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans = df_loans.reset_index(drop = True)\n",
    "df_loans.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1     57302\n",
       "0     16965\n",
       "7     10374\n",
       "2      7268\n",
       "3      7079\n",
       "6      2531\n",
       "4      2395\n",
       "13     1967\n",
       "15     1493\n",
       "18      868\n",
       "14      857\n",
       "20      755\n",
       "19      754\n",
       "5       750\n",
       "16      303\n",
       "11      212\n",
       "8       191\n",
       "10       91\n",
       "9        85\n",
       "12       59\n",
       "17       51\n",
       "Name: ListingCategory (numeric), dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans['ListingCategory (numeric)'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4.0    18320\n",
       "5.0    15410\n",
       "3.0    14274\n",
       "6.0    14030\n",
       "2.0     9795\n",
       "1.0     6935\n",
       "7.0     4633\n",
       "Name: ProsperRating (numeric), dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans['ProsperRating (numeric)'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The values for *Listing Category* and *Prosper Rating* are all valid. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#is there any null listing?\n",
    "df_loans.ListingNumber.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "823"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#look at the remaining duplicate values\n",
    "df_loans.sort_values(['ListingNumber'], inplace=True)\n",
    "df_loans.ListingNumber.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ListingNumber</th>\n",
       "      <th>ListingCreationDate</th>\n",
       "      <th>CreditGrade</th>\n",
       "      <th>Term</th>\n",
       "      <th>LoanStatus</th>\n",
       "      <th>ClosedDate</th>\n",
       "      <th>ProsperRating (numeric)</th>\n",
       "      <th>ProsperScore</th>\n",
       "      <th>ListingCategory (numeric)</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>EmploymentStatus</th>\n",
       "      <th>IsBorrowerHomeowner</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>16108</th>\n",
       "      <td>786407</td>\n",
       "      <td>2013-05-22 19:42:23.417000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Other</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86007</th>\n",
       "      <td>786407</td>\n",
       "      <td>2013-05-22 19:42:23.417000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>60</td>\n",
       "      <td>Current</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Other</td>\n",
       "      <td>Other</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       ListingNumber            ListingCreationDate CreditGrade  Term  \\\n",
       "16108         786407  2013-05-22 19:42:23.417000000         NaN    60   \n",
       "86007         786407  2013-05-22 19:42:23.417000000         NaN    60   \n",
       "\n",
       "      LoanStatus ClosedDate  ProsperRating (numeric)  ProsperScore  \\\n",
       "16108    Current        NaN                      3.0           6.0   \n",
       "86007    Current        NaN                      3.0           4.0   \n",
       "\n",
       "       ListingCategory (numeric) Occupation EmploymentStatus  \\\n",
       "16108                          1      Other            Other   \n",
       "86007                          1      Other            Other   \n",
       "\n",
       "       IsBorrowerHomeowner  \n",
       "16108                False  \n",
       "86007                False  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans[df_loans['ListingNumber']==786407]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> As in the duplicate above, it's hard to tell which of the two lines should be kept. Given that the listings having this issue are less than 1% of our dataset, we will move on and focus on the big picture by exploring the data visually. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(112350, 12)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#shape of final dataset\n",
    "df_loans.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### What is the structure of your dataset?\n",
    "\n",
    "> The dataset is made of 112481 rows and 12 columns. It's a subset of the original ProsperLoanData dataset as we want to focus on fewer variables. We have also dropped a few rows containing an invalid value for the Prosper Score and rows having N/A for both Prosper Rating and Credit Grade.\n",
    "\n",
    "### What is/are the main feature(s) of interest in your dataset?\n",
    "\n",
    "> I kept the variables I was more interested in, thinking of what I would want to analyse:\n",
    "- Time of the loan: when and for how long the loan was open \n",
    "- Credit rating/score: risk scores assigned to each loan\n",
    "- Loan features: category, status\n",
    "- Characteristics of the borrowers: are they home owner, what is their employment status and occupation\n",
    "> My aim would be to find at least one insight in each of the four categories through visualisation.\n",
    "\n",
    "### What features in the dataset do you think will help support your investigation into your feature(s) of interest?\n",
    "\n",
    "> Thanks to the large amount of data and the use of visualisation techniques for analysis, I will try to gather insights on the features mentioned above and the relationships among them. The variables I will use are the following:\n",
    "> - General: \n",
    "    - ListingNumber\n",
    "- Time of the loan:\n",
    "    - ListingCreationDate\n",
    "    - Term\n",
    "    - ClosedDate   \n",
    "- Credit Rating/Score:\n",
    "    - Credit Grade  \n",
    "    - ProsperRating (numeric)\n",
    "    - ProsperScore\n",
    "- Loan features:\n",
    "    - LoanStatus\n",
    "    - ListingCategory   \n",
    "- Borrowers' features:\n",
    "    - Occupation\n",
    "    - EmploymentStatus\n",
    "    - IsBorrowerHomeowner"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Univariate Exploration\n",
    "\n",
    "> Here we will look at each individual variable separately.\n",
    "\n",
    "## Loan term ##\n",
    "### Listing Creation Date ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#change data format to datetime\n",
    "df_loans['ListingCreationDate'] = pd.to_datetime(df_loans['ListingCreationDate'])\n",
    "\n",
    "#create groupby object to get the number of listings each month\n",
    "df_1 = pd.DataFrame(df_loans.set_index('ListingCreationDate').groupby(pd.Grouper(freq='M'))['ListingNumber'].count().reset_index())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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AMBo3u9gnZnh1MALk3xpec0BAABYX/MGrE+21l671JYAAACsuXkD1pur6owkf5Hkc3snttYuWkqrAAAA1tC8Aet+488TJ6a1JA/p2xwAAID1Ne8ogoZiBwCAXeCE08/atHbhGadsY0t2py0DVlV9f2vtj6rqGdPqrbUXLKdZAAAA62fWEaybjz9vMaXWOrcFAABgrW0ZsFprvzPe/bvW2j9M1sZrYwEAADA6aM75XjznNAAAgAPWrO9gfXOSb0ly5IbvYd0yycHLbBgAAMC6mfUdrBslOXScb/J7WJ9K8vhlNQoAAGAdzfoO1luTvLWq/qC19qEkqaqDkhzaWvvUdjQQAABgXcz7HaxfqapbVtXNk3wgyaVVdfoS2wUAALB25g1Y9xiPWD0uyWuSHJvkSUtrFQAAwBqaN2AdUlWHZAhYZ7fWvhDXwQIAALiBeQPW7yT5YIYLD7+tqu6QYaALAAAARrNGEUyStNZelORFE5M+VFUPXk6TAAAA1tOs62B9f2vtjzZcA2vSC5bQJgAAYAEnnH7W1OkXnnHKNrfkwDPrCNbNx5+3mFLzHSwAAIAJs66D9Tvjz+dsrFXVjy2rUQAAAOto3kEuptnstEEAAIAD0iIBq7q1AgAAYBdYJGD5DhYAAMCEWaMIXpvpQaqS3HQpLQIAAFhTswa5mDZ6IAAAAFMscoogAAAAEwQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATgQsAACATpYWsKrqmKp6c1VdUlXvr6qnj9NvXVXnVtVl48/Dx+lVVS+qqsur6j1VdfzEuk4d57+sqk5dVpsBAAAWscwjWF9M8hOttbsnuX+Sp1XVPZI8M8kbW2t3SfLG8XGSPDLJ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\n",
      "text/plain": [
       "<Figure size 864x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "base = sb.color_palette()[0]\n",
    "total = df_loans.shape[0]\n",
    "fig=plt.figure(figsize=(12, 10))\n",
    "\n",
    "sb.barplot(x=\"ListingCreationDate\", y=\"ListingNumber\", data=df_1, color = base)\n",
    "plt.xticks(plt.xticks()[0], (df_1.ListingCreationDate.dt.year.astype(str))+\"-\"+(df_1.ListingCreationDate.dt.month.astype(str)), rotation=90)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> We can see that there is an increasing trend from the end of 2005 to the beginning of 2014: more loans were created in the  last period of the dataset rather than in the initial one. \n",
    "It's also interesting to note that between end of 2008 and end of 2009, which coincides with the period of the financial crisis, almost no new loans were created.\n",
    "\n",
    "### Term ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "ax_term = sb.countplot(data = df_loans, x = 'Term', color = base)\n",
    "for p in ax_term.patches:\n",
    "    height = p.get_height()\n",
    "    ax_term.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Most of the loans (77%) have a 3-year term, 21% have a 5-year term and only 1% are for a year.\n",
    "\n",
    "### Closed Date ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "#same as for the ListingCreationDate\n",
    "df_loans['ClosedDate'] = pd.to_datetime(df_loans['ClosedDate'])\n",
    "df_2 = pd.DataFrame(df_loans.set_index('ClosedDate').groupby(pd.Grouper(freq='M'))['ListingNumber'].count().reset_index())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig=plt.figure(figsize=(12, 10))\n",
    "\n",
    "sb.barplot(x=\"ClosedDate\", y=\"ListingNumber\", data=df_2, color = base)\n",
    "plt.xticks(plt.xticks()[0], (df_2.ClosedDate.dt.year.astype(str))+\"-\"+(df_2.ClosedDate.dt.month.astype(str)), rotation=90)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Here as well we see an increasing trend of number of closed loans. We will create a new variable to look at the distribution of the length of the loans, ie for how long loans were open. \n",
    "\n",
    "### Duration ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_loans['Duration'] = (df_loans['ClosedDate'] - df_loans['ListingCreationDate']).abs().dt.components.days"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_loans['months'] = round(df_loans.Duration/30,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda3/lib/python3.7/site-packages/numpy/lib/histograms.py:754: RuntimeWarning: invalid value encountered in greater_equal\n",
      "  keep = (tmp_a >= first_edge)\n",
      "/anaconda3/lib/python3.7/site-packages/numpy/lib/histograms.py:755: RuntimeWarning: invalid value encountered in less_equal\n",
      "  keep &= (tmp_a <= last_edge)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(data = df_loans, x = 'months');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> We have a bimodal distribution of the loan duration, with peaks at 1 year and 3 years. At a later stage we could compare this with loan status and loan term."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Credit Rating ##\n",
    "### Credit Grade ###\n",
    "This variable was used for loans pre-2009."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "total_credit = df_loans.CreditGrade.notnull().sum()\n",
    "\n",
    "order_credit = df_loans.CreditGrade.value_counts().index.tolist()\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "ax_credit = sb.countplot(data = df_loans, x = 'CreditGrade', color = base, order = order_credit)\n",
    "\n",
    "for p in ax_credit.patches:\n",
    "    height = p.get_height()\n",
    "    ax_credit.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total_credit),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Most of the data is in the middle range ie B-C-D, with fewer loans having very high or very low credit grades. \n",
    "\n",
    "### Prosper Rating ###\n",
    "A measure used after July 2009."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "total_rating = df_loans['ProsperRating (numeric)'].notnull().sum()\n",
    "\n",
    "order_rating = df_loans['ProsperRating (numeric)'].value_counts().index.tolist()\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "ax_rating = sb.countplot(data = df_loans, x = 'ProsperRating (numeric)', color = base, order = order_rating)\n",
    "\n",
    "for p in ax_rating.patches:\n",
    "    height = p.get_height()\n",
    "    ax_rating.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total_rating),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> As we noticed earlier, credit ratings after 2009 (called *Prosper Ratings*) are mainly concentrated in the middle ranges rather than the extreme ones, ie we have fewer loans with rating 1, 2 or 7.\n",
    "\n",
    "### Prosper Score ###\n",
    "A measure used after July 2009."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "total_grade = df_loans['ProsperScore'].notnull().sum()\n",
    "\n",
    "order_grade = df_loans['ProsperScore'].value_counts().index.tolist()\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "ax_grade = sb.countplot(data = df_loans, x = 'ProsperScore', color = base, order = order_grade)\n",
    "\n",
    "for p in ax_grade.patches:\n",
    "    height = p.get_height()\n",
    "    ax_grade.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total_grade),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> This measure as well, which is a custom risk score built by using Prosper historical data, shows concentration in the middle range ie between 4 and 8."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Loan features ##\n",
    "### Loan Status ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "order_status = df_loans.LoanStatus.value_counts().index.tolist()\n",
    "\n",
    "ax = sb.countplot(data = df_loans, x = 'LoanStatus', color = base, order = order_status)\n",
    "plt.xticks(rotation=90)\n",
    "\n",
    "for p in ax.patches:\n",
    "    height = p.get_height()\n",
    "    ax.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> From here we can see that less than 2% of the loans are Past Due, around 4% defaulted and 11% got charged off. With the help of other variables we will be able to get more details on the duration and term of outstanding loans and completed ones. \n",
    "\n",
    "### Listing Category ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "order_listing = df_loans['ListingCategory (numeric)'].value_counts().index.tolist()\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "ax_listing = sb.countplot(data = df_loans, x = 'ListingCategory (numeric)', color = base, order = order_listing)\n",
    "for p in ax_listing.patches:\n",
    "    height = p.get_height()\n",
    "    ax_listing.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> This distribution gives us some interesting insights on the population:\n",
    "- Roughly half of the borrowers asked a loan for Debt Consolidation\n",
    "- 25% either did not give a reason or falls into the \"Other\" loan category\n",
    "- 6% needed money for Home Improvement\n",
    "- 6% for Business"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Borrowers' features ##\n",
    "### Employment Status ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "order_employment = df_loans.EmploymentStatus.value_counts().index.tolist()\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "sb.countplot(data = df_loans, x = 'EmploymentStatus', color = base, order = order_employment);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> The classification of the employment status is not consistent as we have labels such as Employed, Full-time and Part-time, when full-time and part-time are subsets of \"employed\". We can still gather from this chart that most of the loans were given to employed people. The percentage of non-employed, other or retired people is quite low (less than 5%). The fact that the majority of loans is given to employed people was expected. \n",
    "\n",
    "### Is Borrower HomeOwner? ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sb.countplot(data = df_loans, x = 'IsBorrowerHomeowner', color = base);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> It looks like our data is split almost 50-50 between homeowners and not. \n",
    "\n",
    "### Occupation ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA4sAAAHUCAYAAACTVL1oAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzs3XuYXWV59/HvnYQAIXIGJSQYQpCSAEkgnIS2gq0QrEEULFoBBYUqWEBfAQ8VKiJYD7VIpWLl5IGgiAQsBCIqigdCkCgQtaEcEyhnhKKQEu73j7Um2TNrz2SSzMxam3w/1zXXzH722nvumWT2Xr+1nnU/kZlIkiRJktRqWN0FSJIkSZKax7AoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKliRN0FDLXNN988x48fX3cZkiRJklSL22677fHM3GJl2611YXH8+PHMnz+/7jIkSZIkqRYRcX9/tnMaqiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw2If5syZww477MDEiRM555xzKvd/4QtfYNKkSeyyyy68/vWv5/77V1wnOnz4cKZOncrUqVOZOXPmUJYtSZIkSWtsreuG2l/Lli3j+OOPZ+7cuYwdO5bdd9+dmTNnMmnSpOXbTJs2jfnz5zNq1CjOP/98TjnlFC6//HIA1l9/fRYsWFBX+ZIkSZK0Rjyz2It58+YxceJEJkyYwMiRIzn88MOZPXt2t232228/Ro0aBcBee+3F4sWL6yhVkiRJkgacYbEXS5YsYdy4cctvjx07liVLlvS6/de+9jVmzJix/Pbzzz/P9OnT2WuvvbjqqqsGtVZJkiRJGmhOQ+1FZlbGIqLttt/4xjeYP38+N9100/KxBx54gDFjxnDPPfew//77s/POO7PddtsNWr2SJEmSNJA8s9iLsWPH8uCDDy6/vXjxYsaMGVPZ7gc/+AFnnXUWV199Neuuu+7y8a5tJ0yYwOte9zpuv/32wS9akiRJkgaIYbEXu+++O4sWLeLee+9l6dKlzJo1q9LV9Pbbb+e4447j6quvZsstt1w+/tRTT/HCCy8A8Pjjj/Ozn/2sW2McSZIkSWq6tX4a6m4fvrTX+4ZNO4Qdd9ubfOklNtv5Lzjy4tt46OaPMepV49l44q4s+vZn+NPjjzNln78CYOSGm7LdISfzv0sW8cDci4kIMpMtd3sDR1w0H5jf9vvc9tkjB+NHkyRJkqTVttaHxb5sNGEKG02Y0m1szL5vWf719m87te3jRm+9PZPeddag1iZJkiRJg8lpqJIkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkikELixExLiJ+FBG/jYi7IuLEcvyMiFgSEQvKj4NaHvORiLg7In4fEQe0jB9Yjt0dEae1jG8bEbdExKKIuDwiRg7WzyNJkiRJa5PBPLP4IvChzNwR2As4PiImlff9S2ZOLT+uBSjvOxyYDBwIfDkihkfEcODfgBnAJODtLc/zmfK5tgeeAo4ZxJ9HkiRJktYagxYWM/PhzPxV+fWzwG+Brft4yMHArMx8ITPvBe4G9ig/7s7MezJzKTALODgiAtgfuKJ8/CXAmwfnp5EkSZKktcuQXLMYEeOBacAt5dAJEfGbiLgwIjYpx7YGHmx52OJyrLfxzYCnM/PFHuPtvv+xETE/IuY/9thjA/ATSZIkSdLL26CHxYgYDXwXOCkznwHOB7YDpgIPA5/v2rTNw3M1xquDmRdk5vTMnL7FFlus4k8gSZIkSWufEYP55BGxDkVQ/GZmXgmQmY+03P9V4PvlzcXAuJaHjwUeKr9uN/44sHFEjCjPLrZuL0mSJElaA4PZDTWArwG/zcwvtIxv1bLZIcCd5ddXA4dHxLoRsS2wPTAPuBXYvux8OpKiCc7VmZnAj4BDy8cfBcwerJ9HkiRJktYmg3lmcR/gCOCOiFhQjn2UopvpVIopo/cBxwFk5l0R8W1gIUUn1eMzcxlARJwAXA8MBy7MzLvK5zsVmBURnwJupwinkiRJkqQ1NGhhMTNvpv11hdf28ZizgLPajF/b7nGZeQ9Ft1RJkiRJ0gAakm6okiRJkqTOYliUJEmSJFUYFiVJkiRJFYZFSZIkSVKFYVGSJEmSVGFYlCRJkiRVGBYlSZIkSRWGRUmSJElShWFRkiRJklRhWJQkSZIkVRgWJUmSJEkVhkVJkiRJUoVhUZIkSZJUYViUJEmSJFUYFiVJkiRJFYZFSZIkSVKFYVGSJEmSVGFYlCRJkiRVGBYlSZIkSRWGRUmSJElShWFRkiRJklRhWJQkSZIkVRgWJUmSJEkVhkVJkiRJUoVhUZIkSZJUYViUJEmSJFUYFiVJkiRJFYZFSZIkSVKFYVGSJEmSVGFYlCRJkiRVGBYlSZIkSRWGRUmSJElShWFRkiRJklRhWJQkSZIkVRgWJUmSJEkVhkVJkiRJUoVhUZIkSZJUMWhhMSLGRcSPIuK3EXFXRJxYjm8aEXMjYlH5eZNyPCLi3Ii4OyJ+ExG7tjzXUeX2iyLiqJbx3SLijvIx50ZEDNbPI0mSJElrk8E8s/gi8KHM3BHYCzg+IiYBpwE3Zub2wI3lbYAZwPblx7HA+VCES+B0YE9gD+D0roBZbnNsy+MOHMSfR5IkSZLWGoMWFjPz4cz8Vfn1s8Bvga2Bg4FLys0uAd5cfn0wcGkWfglsHBFbAQcAczPzycx8CpgLHFjet2Fm/iIzE7i05bkkSZIkSWtgSK5ZjIjxwDTgFuCVmfkwFIES2LLcbGvgwZaHLS7H+hpf3Ga83fc/NiLmR8T8xx57bE1/HEmSJEl62Rv0sBgRo4HvAidl5jN9bdpmLFdjvDqYeUFmTs/M6VtsscXKSpYkSZKktd6ghsWIWIciKH4zM68shx8pp5BSfn60HF8MjGt5+FjgoZWMj20zLkmSJElaQ4PZDTWArwG/zcwvtNx1NdDV0fQoYHbL+JFlV9S9gD+U01SvB94QEZuUjW3eAFxf3vdsROxVfq8jW55LkiRJkrQGRgzic+8DHAHcERELyrGPAucA346IY4AHgMPK+64FDgLuBv4IvBsgM5+MiDOBW8vtPpmZT5Zfvw+4GFgfuK78kCRJkiStoUELi5l5M+2vKwR4fZvtEzi+l+e6ELiwzfh8YKc1KFOSJEmS1MaQdEOVJEmSJHUWw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKmiX2ExIm7sz5gkSZIk6eVhRF93RsR6wChg84jYBIjyrg2BMYNcmyRJkiSpJn2GReA44CSKYHgbK8LiM8C/DWJdkiRJkqQa9RkWM/NfgX+NiA9k5peGqCZJkiRJUs1WdmYRgMz8UkS8Fhjf+pjMvHSQ6pIkSZIk1ahfYTEivg5sBywAlpXDCRgWJUmSJOllqL9LZ0wH9snM92fmB8qPf+jrARFxYUQ8GhF3toydERFLImJB+XFQy30fiYi7I+L3EXFAy/iB5djdEXFay/i2EXFLRCyKiMsjYmT/f2xJkiRJUl/6GxbvBF61is99MXBgm/F/ycyp5ce1ABExCTgcmFw+5ssRMTwihlM00pkBTALeXm4L8JnyubYHngKOWcX6JEmSJEm96Nc0VGBzYGFEzANe6BrMzJm9PSAzfxIR4/v5/AcDszLzBeDeiLgb2KO87+7MvAcgImYBB0fEb4H9gXeU21wCnAGc38/vJ0mSJEnqQ3/D4hkD+D1PiIgjgfnAhzLzKWBr4Jct2ywuxwAe7DG+J7AZ8HRmvthm+4qIOBY4FmCbbbYZiJ9BkiRJkl7W+tsN9aYB+n7nA2dSNMc5E/g8cDQr1m/s9m1pP002+9i+rcy8ALgAYPr06b1uJ0mSJEkq9Lcb6rOsCGMjgXWA5zJzw1X5Zpn5SMtzfhX4fnlzMTCuZdOxwEPl1+3GHwc2jogR5dnF1u0lSZIkSWuoXw1uMvMVmblh+bEe8FbgvFX9ZhGxVcvNQyga5wBcDRweEetGxLbA9sA84FZg+7Lz6UiKJjhXZ2YCPwIOLR9/FDB7VeuRJEmSJLXX32sWu8nMq1qXsWgnIi4DXgdsHhGLgdOB10XEVIqzlPcBx5XPd1dEfBtYCLwIHJ+Zy8rnOQG4HhgOXJiZd5Xf4lRgVkR8Crgd+Nrq/CySJEmSpKr+TkN9S8vNYRTrLvZ57V9mvr3NcK+BLjPPAs5qM34tcG2b8XtY0TFVkiRJkjSA+ntm8U0tX79IcVbw4AGvRpIkSZLUCP3thvruwS5EkiRJktQc/WpwExFjI+J7EfFoRDwSEd+NiLGDXZz6Z86cOeywww5MnDiRc845p3L/T37yE3bddVdGjBjBFVdcsXx8wYIF7L333kyePJlddtmFyy+/fCjLliRJktRg/QqLwEUUHUvHAFsD15RjqtmyZcs4/vjjue6661i4cCGXXXYZCxcu7LbNNttsw8UXX8w73vGObuOjRo3i0ksv5a677mLOnDmcdNJJPP3000NZviRJkqSG6u81i1tkZms4vDgiThqMgrRq5s2bx8SJE5kwYQIAhx9+OLNnz2bSpEnLtxk/fjwAw4Z1Pzbwmte8ZvnXY8aMYcstt+Sxxx5j4403HvzCJUmSJDVaf88sPh4R74yI4eXHO4EnBrMw9c+SJUsYN27c8ttjx45lyZIlq/w88+bNY+nSpWy33XYDWZ4kSZKkDtXfsHg08Dbgf4CHgUMBm940QGZ1BZOIWKXnePjhhzniiCO46KKLKmcfJUmSJK2d+jsN9UzgqMx8CiAiNgU+RxEiVaOxY8fy4IMPLr+9ePFixowZ0+/HP/PMM7zxjW/kU5/6FHvttddglChJkiSpA/X3NNIuXUERIDOfBKYNTklaFbvvvjuLFi3i3nvvZenSpcyaNYuZM2f267FLly7lkEMO4cgjj+Swww4b5EolSZIkdZL+nlkcFhGb9Diz2N/HagDs9uFLe71v2LRD2HG3vcmXXmKznf+CIy++jYdu/hijXjWejSfuynMP38M9s89l2fPP8Y3Lr2Cdvz+RSe8+mycW/oz7f3wTv7jjbj766X8B4NUz3sOoLV/d9vvc9tkjB+VnkyRJktQ8/Q18nwd+HhFXAElx/eJZg1aVVslGE6aw0YQp3cbG7PuW5V9vsNUEdv77L1Yet9mkfdhs0j6DXp8kSZKkztOvsJiZl0bEfGB/IIC3ZObClTxMkiRJktSh+j2VtAyHBkRJkiRJWgu4ToIkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqSKQQuLEXFhRDwaEXe2jG0aEXMjYlH5eZNyPCLi3Ii4OyJ+ExG7tjzmqHL7RRFxVMv4bhFxR/mYcyMiButnkSRJkqS1zWCeWbwYOLDH2GnAjZm5PXBjeRtgBrB9+XEscD4U4RI4HdgT2AM4vStgltsc2/K4nt9LkiRJkrSaBi0sZuZPgCd7DB8MXFJ+fQnw5pbxS7PwS2DjiNgKOACYm5lPZuZTwFzgwPK+DTPzF5mZwKUtzyVJkiRJWkNDfc3iKzPzYYDy85bl+NbAgy3bLS7H+hpf3Ga8rYg4NiLmR8T8xx57bI1/CEmSJEl6uWtKg5t21xvmaoy3lZkXZOb0zJy+xRZbrGaJkiRJkrT2GOqw+Eg5hZTy86Pl+GJgXMt2Y4GHVjI+ts24JEmSJGkADHVYvBro6mh6FDC7ZfzIsivqXsAfymmq1wNviIhNysY2bwCuL+97NiL2KrugHtnyXJIkSZKkNTRisJ44Ii4DXgdsHhGLKbqangN8OyKOAR4ADis3vxY4CLgb+CPwboDMfDIizgRuLbf7ZGZ2Nc15H0XH1fWB68oPSZIkSdIAGLSwmJlv7+Wu17fZNoHje3meC4EL24zPB3ZakxolSZIkSe01pcGNJEmSJKlBDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqDIuSJEmSpArDoiRJkiSpwrAoSZIkSaowLEqSJEmSKgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKnCsChJkiRJqjAsSpIkSZIqagmLEXFfRNwREQsiYn45tmlEzI2IReXnTcrxiIhzI+LuiPhNROza8jxHldsvioij6vhZJEmSJOnlqM4zi/tl5tTMnF7ePg24MTO3B24sbwPMALYvP44FzociXAKnA3sCewCndwVMSZIkSdKaadI01IOBS8qvLwHe3DJ+aRZ+CWwcEVsBBwBzM/PJzHwKmAscONRFq//mzJnDDjvswMSJEznnnHMq97/wwgv87d/+LRMnTmTPPffkvvvu63b/Aw88wOjRo/nc5z43RBVLkiRJa6+6wmICN0TEbRFxbDn2ysx8GKD8vGU5vjXwYMtjF5djvY1XRMSxETE/IuY/9thjA/hjqL+WLVvG8ccfz3XXXcfChQu57LLLWLhwYbdtvva1r7HJJptw9913c/LJJ3Pqqad2u//kk09mxowZg16roVaSJEmqLyzuk5m7UkwxPT4i/qKPbaPNWPYxXh3MvCAzp2fm9C222GLVq9UamzdvHhMnTmTChAmMHDmSww8/nNmzZ3fbZvbs2Rx1VHHp6aGHHsqNN95IZvFPetVVVzFhwgQmT548qHV2UqiVJEmSBlMtYTEzHyo/Pwp8j+Kaw0fK6aWUnx8tN18MjGt5+FjgoT7G1UBLlixh3LgV/1xjx45lyZIlvW4zYsQINtpoI5544gmee+45PvOZz3D66acPep2dEmolSZKkwTbkYTEiNoiIV3R9DbwBuBO4GujqaHoU0LWHfjVwZNkVdS/gD+U01euBN0TEJmVjmzeUY2qgrjDVKiL6tc3pp5/OySefzOjRowetvi6dEmolSZKkwTaihu/5SuB7ZVAYAXwrM+dExK3AtyPiGOAB4LBy+2uBg4C7gT8C7wbIzCcj4kzg1nK7T2bmk0P3Y2hVjB07lgcfXHGJ6eLFixkzZkzbbcaOHcuLL77IH/7wBzbddFNuueUWrrjiCk455RSefvpphg0bxnrrrccJJ5ww4HV2SqiVJEmSBtuQn1nMzHsyc0r5MTkzzyrHn8jM12fm9uXnJ8vxzMzjM3O7zNw5M+e3PNeFmTmx/LhoqH8W9d/uu+/OokWLuPfee1m6dCmzZs1i5syZ3baZOXMml1xSNMS94oor2H///YkIfvrTn3Lfffdx3333cdJJJ/HRj350UIIirFqoBSqh9pRTTmH8+PF88Ytf5NOf/jTnnXfeoNQJq9+IZ968eUydOpWpU6cyZcoUvve97w1ajZ1UpyRJkrqr48yiXsZ2+/Clvd43bNoh7Ljb3uRLL7HZzn/BkRffxkM3f4xRrxrPxhN35aUX1+W+H/2Kizd5JcPX24Bt/+b9led76Ge/ZvjIdbnskd6/z22fPXK1628NtVtvvTWzZs3iW9/6VrdtukLt3nvvXQm1Xc444wxGjx49aKG2qxHP3LlzGTt2LLvvvjszZ85k0qRJy7dpbcQza9YsTj31VC6//HJ22mkn5s+fz4gRI3j44YeZMmUKb3rTmxgxYuBfDjqlzjlz5nDiiSeybNky3vOe93Daaad1u/+FF17gyCOP5LbbbmOzzTbj8ssvZ/z48cybN49jjy0aOmcmZ5xxBocccsiA1ydJklQHw6KGzEYTprDRhCndxsbs+5blXw8bMZIJM/sOV2P2WfMd8b4CLXRGqG1txAMsb8TTGsJmz57NGWecARSNeE444QQyk1GjRi3f5vnnn69Msx1InVBnpwRaMNRKkqShZViUemhKqO1Lu0Y8t9xyS6/btDbi2Xzzzbnllls4+uijuf/++/n6178+aOGmE+rshEALnRVqJUnSy0Nd6yxKWgNr0ogHYM899+Suu+7i1ltv5eyzz+b5559fa+tckw64ALfccguTJ09m55135t///d8HLYCtybIuo0aNWl7XYIda8DpVSZJeLgyLUgdak0Y8rXbccUc22GAD7rzzzrW2zk4ItNA5obbrDOh1113HwoULueyyy1i4cGG3bVrPgJ588smceuqpAMvPgC5YsIA5c+Zw3HHH8eKLLw5KnWColSRpZQyLUgdak+6y99577/Id8Pvvv5/f//73jB8/fq2tsxMCLXROqO2UM6CdFGolSaqLF6xIDTVYjXieuOtnPDLv+8SwERDBVnsfxgGfubbX77OyRjydUmdv1qQD7r333su4ceMYMWLEoAfvNVmrtFVrqJ0+ffqA19kJ16lC51yrKklSnQyLUoda3UY8m03eh80m7zPo9XVpSp2DsazLUAVa6JxQO1BnQH/7299y1FFHMWPGDNZbb70Br7NTQi2sfhfcuXPnctppp7F06VJGjhzJZz/7Wfbff/9Bq1OS9PJjWJS01mtKoO3LiBEjOO+88zjggANYtmwZRx99NJMnT+YTn/gE06dPZ+bMmRxzzDEcccQRTJw4kU033ZRZs2YBcPPNN3POOeewzjrrMGzYML785S+z+eabD0qdnXIGtFNC7Zp0wd1888255pprGDNmDHfeeScHHHBA5TrXgWSolaSXH8OiJDXIyqb1vuLgjwNw5dNw5YcvBSbynz99mn/6afm48W9io/FvYhlw2Pk3AzcDsN5Bpy5/jjN/9gxn/mxw1gDtlDOgnRJq12S67LRp05ZvM3nyZJ5//nleeOEF1l133QGvs1NCrYFWklaNDW4kSQOm9QzojjvuyNve9rblZ0CvvvpqAI455hieeOIJJk6cyBe+8IXlnUhvvvlmpkyZwtSpUznkkEMG9QxoJzRfgjXvgtvlu9/9LtOmTRuUoAhr1tho2rRpy4N6a6gdaGvS1Kgr0N5xxx1ccsklHHHEEQNeX6vV7dQ7d+5cdtttN3beeWd22203fvjDHw5qnZJe/jyzKElaZZ1yBrTTr1Vd0+myAHfddRennnoqN9xww2rXsTJreg1ol8EMtZ6llaRVZ1iUJK2VOuFa1TWdLrt48WIOOeQQLr30UrbbbrtBq7MTQm0nBFronFArae3gNFRJkhpqTabLPv3007zxjW/k7LPPZp99Bjfcrul6pUMRagcy0H7lK18Z+AJLnTL1WNLawTOLkiTVaLDWKn34F7N5ZOHvOOJ9H+SI930QgImHfph1Ntiw7fepq7HRUIVaz9JK0qozLEqS1GCrO112q70PZqu9Dx6wOjo91HZCoIXOCbWS1g6GRUmStMY6IdQOVaCFl3+oBZcikdYGhkVJkrRWaEqghWaE2jWZetxJXVtXN9Q+8cQTHHroodx66628613v4rzzzhu0GqWmssGNJElSg2w0YQqTj/lndnrv59hqr6Kh0Zh938LGE3cFVoTaye/5LH/2zjNYd+MtgSLUTj3pq+x41JnLP/o6+7kmOmFtTViz9TXXW289zjzzTD73uc8NSm09re76mk888QT77bcfo0eP5oQTqgc71tY6NTAMi5IkSVolndK1dU1C7QYbbMC+++7LeuutNyi1teqUUNspdRpoB45hUZIkSatkbVuKZLB1SqjthDo7JdBCZ4Raw6IkSZJWSSesrQkDE2qHQqeE2k6osxMCLXROqDUsSpIkaZW0dm1dunQps2bNYubMmd226eraCnTEUiRQDbVDpVNCbSfU2QmBFjon1NoNVZIkSW11etfWNVmKZCit6fqa1rlCJwRaaB9qb7nlll63aQ21m2+++ZDVaViUJEnSKmvSUiS9GTFiBOeddx4HHHAAy5Yt4+ijj2by5Ml84hOfYPr06cycOZNjjjmGI444gokTJ7Lpppsya9as5Y8fP348zzzzDEuXLuWqq67ihhtu6LY8yEDplFDbCXV2QqCFzgm1hkVJkiR1tL7OgAK84uCPA3Dl03Dlhy8FJvKfP32af/pp+bjxb2Kj8W9iGXDY+TcDNwOw2WGfZLOW5zniovnA/LbfY2VnQFdW4+qeqQW484IPsWzpn8hlL3LBJd9i4qEfZv3Nt35Z19mbTgi00Dmh1rAoSZIk1Wx1z9QC7HTs5we1tlZNr9OzyQPLsChJkiSpozThbDKs2ZnaoTpL2586e2NYlCRJkqQh1vSztODSGZIkSZKkNgyLkiRJkqQKw6IkSZIkqcKwKEmSJEmqMCxKkiRJkioMi5IkSZKkCsOiJEmSJKmi48NiRBwYEb+PiLsj4rS665EkSZKkl4OODosRMRz4N2AGMAl4e0RMqrcqSZIkSep8HR0WgT2AuzPN3uplAAAgAElEQVTznsxcCswCDq65JkmSJEnqeJ0eFrcGHmy5vbgckyRJkiStgcjMumtYbRFxGHBAZr6nvH0EsEdmfqDHdscCx5Y3dwB+P8ClbA48PsDPOdA6oUawzoFmnQPLOgdOJ9QI1jnQrHNgWefA6YQawToH2tpc56szc4uVbTRigL/pUFsMjGu5PRZ4qOdGmXkBcMFgFRER8zNz+mA9/0DohBrBOgeadQ4s6xw4nVAjWOdAs86BZZ0DpxNqBOscaNa5cp0+DfVWYPuI2DYiRgKHA1fXXJMkSZIkdbyOPrOYmS9GxAnA9cBw4MLMvKvmsiRJkiSp43V0WATIzGuBa2suY9CmuA6gTqgRrHOgWefAss6B0wk1gnUONOscWNY5cDqhRrDOgWadK9HRDW4kSZIkSYOj069ZlCRJkiQNAsOiJEmSJKnCsKjaRMTwiPhs3XW8XJS/z7fWXUd/RMTX+zMmqXNFxLCIeG3ddfRHuW7zSsf08lC+X15Sdx0vF1EYt/IttTIRsWlfH7XU5DWLqyYihgF7ZebP667l5SAifgi8PjvkP2JEvBrYPjN/EBHrAyMy89m66+oSET/NzD+vu46ViYhfZeauLbeHA3dk5qQay6qIiHUz84WVjTVBuVM+npbGZZl5aW0FrUREbAAcArw9M99Ydz2dIiLe0tf9mXnlUNXSHxHxi8zcu+46Vqbna1JvY3WKiLEUS4T9OTAG+BNwJ/CfwHWZ+VKN5XWciLgBeGNm/l/dtfSlU/Y7I+K2zNyt7jpWJiJeCXwaGJOZMyJiErB3Zn6t5tIAiIh7gQSizd2ZmROGuKTO74Y61DLzpYj4PNDYN7+IuIbiP1pbmTlzCMtZmduB2RHxHeC5rsGm7fAARMR7gWOBTYHtgLHAvwOvr7OuHq6PiJOAy+n++3ymvpJWiIiPAB8F1o+IrpoCWEozO5L9Aui5s9hurFblWdntgAXAsnI4gUaFxXI93IOAdwAHAt+l+BtqhIj4YF/3Z+YXhqqWPrypj/sSaNpr5w3ljIcrm3hQMCJmUPyf3Doizm25a0PgxXqqqoqIi4Ctge8DnwEeBdYDXkPxt/SxiDgtM39SX5UrRMQWwHupHsA6uq6a2rgH+GlEzKb7++W5vT9k6HXCfmfplxGxe2beWnchK3ExcBHwsfL2f1HsMzUiLGbmtnXX0JNhcfU0+s0P+FzdBayCTYEngP1bxpq4wwNwPLAHcAtAZi6KiC3rLaniuPLzh1hxZCqBbWqrqEVmng2cHRFnZ+ZH6q6nNxHxKoods/UjYhorjvBtCIyqrbDeTQcmNfT1iIj4a+DtwAHAj4CvA3tk5rtrLazqFXUXsDIN/J2tzAeBDYBlEfEnytekzNyw3rKWewiYD8wEbmsZfxY4uZaK2vt8Zt7ZZvxO4MryQEwjXudLs4GfAj9gxQGspnkMmEvxmt7E1/VWTd/vBNgP+PuIuI8ifHf9re9Sa1VVm2fmt8uD111rtjfy/2hEbAJsT3FgCIA6Dgg5DXU1RMSzlG9+FNNAmvbmp0EQEbdk5p4RcXtmTouIEcCvGvhC2BEiYmvg1XQ/6tyUo+JHAe+iCGG3siIsPgtc3LQz3+WZ+X/IzIfrrqWdiHiJYsfxXZl5bzl2Tx3TaTpdRLwzM7/R21nQhpz97DgRsU7TpyN2kohYkJlT666jP5p6aUGrTtjvLC/TqcjM+4e6lr5ExI+BtwJzM3PXiNgL+Exm/mW9lXUXEe8BTqSYxbYA2Av4RWbu3+cDB4FnFldDZjb+6DNARGwPnA1MovtRicbsoEXEa4DzgVdm5k4RsQswMzM/VXNp7dwUEV1TKP8aeD9wTc01dVNeR3ki8OrMfF9ETKS4xvK6mkvrJiLOobj2ZiHdp002Iixm5iXAJRHx1sz8bt319MPmwMKImAcs3+lp0JTz3Sj+vX8QEfcAs4Dh9ZbUu4hYDzgGmEz3184mTKHboPzcEe9DABExE/iL8uaPM/P7ddbTiz0i4gxWHMDq2hlvxPtlRPyI3i8vycxs0uUQAN+PiIMy89q6C+lNROxBMfVwI2CbiJgCvCczP1BvZVWdsN+ZmfdHxL4U+xwXlVORR9ddVxsfBK4GtouInwFbAIfWW1JbJwK7A7/MzP0i4s+Af6qjEM8sroaICODvgG0z88yyA9RWmTmv5tK6iYibgdOBf6G4zuXdFP/mp9daWIuIuAn4MPCVzJxWjt2ZmTvVW1lVeZH5McAbKHYkrgf+o0lTQiLiMuAO4B1l+B4F/Kzrd9sUEfF7YJcOOJp7IsW1Dc8CX6W4VvG0zLyh1sJ6iIi2R0Qz86ahrmVlImIfiimpb6U4Wvq9zGzU9arlmdrfUVxb+UmK1/vfZuaJtRbWgcoDQ7sD3yyH3g7clpmn1VdVVUT8jmLa6W20TJvMzCdqK6pFRLRrHLIXcArwaGbuPsQl9anlTNhSoOuMbdPOhP0S+Fvgqg7Y/2j8fmdEnE4xG2eHzHxNRIwBvpOZ+9RcWkU5M2wHin253zdxVkFE3JqZu0fEAmDPzHyhrjP2hsXVEBHnAy8B+2fmjuWc4hsa+GJ9W2buFhF3ZObO5VijumW2/DHc3vJi3cjpKxFxCHBtkwNORMzPzOlN/31GxHXAYZn5v3XX0peI+HVmTomIAyiuWf1H4KJsUIfELmWHt67XoHmZ+Wid9axMefDlr4HDm3YdXstU899k5i4RsQ5wfR3Tf3rT8LOfy0XEb4CpWXbqjKLz8e1Nm77fdZlB3XX0R3lw6B+BdYFPN23mSKeIiHmZuUeP98tfZ+aUumvrqRP2O8tQM43i8pyu3+dvGvi3fjzwzcx8ury9CUVX7i/XW1l3EfE9ipM8J1H09XgKWCczDxrqWpyGunr2LOc53w6QmU+VF5c3zfPlDtmiiDgBWAI0rSHL4xGxHeX0mog4FGjkdVcUDRC+GBE/oZhGd31mNqZbXmlpuRPZ9fvcluLIbtP8EVgQETfSfdrkP9RXUltd1yoeRBESf10e4W2UiHgb8FngxxQ1fykiPpyZV9RaWCmKRlAfBSZSnPk+O4sOvdeXH03TdZT56YjYCfgfiq6OTfJ1irOfB9By9rPWinq3MfBk+fVGdRbShx9Fse7vlXR/TfpVfSV1Vx60+kfgeeCszPxRzSX1qQOmHz9YTkXN8iDGByg6YzZRJ+x3Ls3MjIiu/Y8NVvaAmrw3M/+t60b5u3wv0KiwmJmHlF+eUU5D3wiYU0cthsXV83/lC0vXH8QWFEd8muYkig5f/wCcSXFk4qhaK6o6nmLJhD+LiCXAvcA76y2pvcx8d3mGYQbF9LQvR8TczHxPzaW1+iTFi8nYKBYc/kuKsw9Nc3X50XS3RbEW17bARyLiFTTzb/1jwO5dZxPL16QfAI0IixRLeNwGfAn4G+BcigZCTXVBebT54xT/T0cDn6i3pIqJmXlYRBycmZdExLdoZvA+G7i93NkJivDQxE7IXWcVp7eMJd07ddcmIm6luLbqsxTL9xARy2c4NCnUQtvpxydGxL4Nm378PorXom2ARyheM99Xa0W964T9zm9HxFeAjcvwdTTF5RtNMywiousSovL32rTgTRSNd+7KzGcz86Zy/2MaZUf+Ia3FaairLiL+jmKe+67AJRQXxn48M79Ta2EdrDwCNSwbtMB9b8rAeCDF9IA/z8wtai6pm/JN5LUUO2Y/b/p0xCYrz8xPBe7JzKcjYjNg68z8Tc2lddM61by8PQz4detYnXpOhY6GLXbeiVqm0P2EotnW/1BMP25EQ5ZWEbEV3adI/0+d9XSiKDo4du2w9VywO5s0RRo6Y/pxRGyamU+ufMv6dcp+ZxTN/5b3dcjMuTWXVFHOIBhPscZvAn8PPJiZH6qzrp7Ks8i7toTaYcD8Ot47PbO4GjLzmxFxG8Vi7AG8OTMbN/0nik6jH6a6PEFj3lQiYmPgSMqFe7tm+DVwOiIRcSBFR8f9KKb7/Qfwtjpr6sVwiqm8I4CJETExM39ec03dRAd06i19m6LBzQJY3uyiEQ0vepgTEdcDl5W3/xZoUhfCKM/Ude3gDm+93bQdtoj4NPDPPa5p+VBmfrzeyrppd/bzH+stqVd7A/tS7JgNB75XbzlV5TW/nwbGZOaMiJgE7J2ZTVmo+3V117Aamj79+NYomq1dTrF+YWMPVnfCfmd52ctPuwJiRKwfEeMz8756K6s4lWJN6vdR/C5voNifa5rlZz8BMvOlKBrz1FJIHd+345VHyV5J9xD2QH0VVUXErymOnPTs7nZbrw8aYhHxc+CXFNcxLZ9SkcXSBY0SEbMorlW8rqlNbsqd3HdSXLvU9fvMOi6I7kt0QKdegIj4K4ra9gK+Q7HG4u/qraq9KBZs3ofize8nmdmYHfIoFml+ie5nQ7pk0w4StDa8aBlr1NnQiNg2yzUr+xqrW0R8meJa1dYDGf+dmcfXV1VVFE23LgI+Vja1GkFxJqwRZ+d7U57JOSUz/7ruWlpFxNuBc4Bu048zc1athfUQEa+lOAg8k+Kg4KwG1jgM+E02sEtrq4iYD7w2M5eWt0dSdGNvTBOeThIRV1KcmDi/HHo/sF9mvnnIazEsrrqI+ADFju4jFCGsaz2mxkyvgBXdUOuuoy9N2wHrdOVR0imZ+XzdtfQlOqBTb6uI2Iii5f/HgAcprsP4Rjaw3bbWXDmFbveug0JRrF86PzMn11vZCu1eO5v4mh8RdwE79ZhKdUeTfpfQ/M7cEbE/xcHfMcBVFGdBL6XY/zgrM6+ssby2WqYfB3BLk6cfR8SmwBeBv8vMxq0BGxHfpAjbjTop0ard30s0qLtsRHw7M98WEXfQZs3SBu7Db0lxTe3+FPXeCJxUx6VFTkNdPSdSrCPTxOlora6JiPdTTPlp7e7WpClfXy8vhP4+Da0xIm7OzH2jWDeq9QWm6yBBY9aNomgQNKzuIvqhEzr1AlBep/hO4AjgdoqGDftSNIt6XX2Vdc7/zSi6OL4ie3RnjYh3AI818LqWbwA3RsRFFL/XoymuE6pdFAszTwY2ioi3tNy1IS1Tuhvk9xQNRO4vb48DGnXNb+m58m+9K9TuBfyh3pK6+TxwLEVzmxkUM3L+MTP/tdaqeoiIP8vM37U031lcfh4TEWOa1IgnIkYDB1OcWdwRmE1xvX8TbQXcFRHzgOe6BjNzZn0lVTwWETMz82qAiDgYeLzmmlp1rZP7N7VW0U9lKDy87jrAM4urJYqubn+dzVs2oZuIaDcdqVFTvqJY7+Ys4GlaLt5vUo2dICL+heL3Nw7YhaKrW2v4/mBNpbUVEbtTTJXdmKJT74YU14gNeZevvpTTQP6MYpmCizPz4Zb75mfm9F4frOWiWPz6TZn5WI/xVwHfy8y966msdxExgxXXB92QmY3oNFrugL2ZYtpca0fhZymm0DXi+uSIuIbiNWkjirNL88rbe1I03vqrGsurKMPNl4CdgDspOo8e2pRmVj3PJEfEf2fmdnXW1E5EXJCZx5b7ST01qhFPOT3+GuDbmfnTmsvpUxRra1Zk5k1DXUtvolgG7ZsUZ7+DYhbOkZl5d62F9RARJ1P8my+pu5Z2IuKUzPzniPgS7c+ADnlPD8PiKoiIrh3uycAOwH/SfYf8C3XU1cki4r8p1g9q0tGntiLi65l5xMrG6hARfS6P0ZQmDV0i4rDs0cWt3VjdImL/zPxh3XWsTPkmvTgzX4iI11EcMLi0q0FL3aKPhZn7uk+9i4i9M/MXddfRm952brs0aSe3S3md4g4UO7q/b9I084i4B/h/LUOfa73dtGmoEbFez8sh2o3VKSKGZdmttROUTZhauwo3stN5ecY2mtowKCJOp2hO+CRFH4orMvOReqtaISLelJnXRETbpe7q6OlhWFwF5X+w3mRmfnLIiumHKJZ4eB8ti+ICX2nYG+DVwOGZ+ce6a1mZNkd2R1BcdD6pxrK6iYj1KBbG7WpXPgwY2aQ3aOj1eqvGXL/aY3pfRQN3zBZQrA83nmKtvasppso3orFRRPwXMKnnbIzyNWphZm5fT2Xddcq0XqCr2/X5wCszc6eI2AWYmZmfqrm0jhRF07o3Unbm7hpvykHgckp0bzIzjx6yYvqhya/xEfH5zPxQRHyP9mdu+nz9r0NEvI1ijc0fU7we/Tnw4Z5T++sQEe/MzG+0nFDppil/Qz2Vr5l/C7yV4mBro2Y7NInXLK6CzPwn6P2sSD1V9el8YB3gy+XtI8qxJi0ivwxYUE5ZaT1L25ilMyLiI8BHgfUj4pmuYWApcEFthbX3I4o1jrqO6G1AER4acR1GObXvIGDriDi35a4NgSZN635TH/cl0KiwCLyUmS9GxCHAFzPzS1Gs0dQUVwJfjYgTMvM5gCjWVj2XBv0uM3Pf8vMr6q6lH75KsTTSVwAy8zcR8S2gUWGxvPbvSxTXhI2kWDrjuSYF79I1wPP06MzdFJn57rpr6I9yavnWFO+X02B5B+QNgVG1Fdbd5eXn82qtYtV8jKLp1qMAUayn/AOg9rDIin/XTnjdbPUoxfq0T9DAngnlAcH/R/UA1pBP5TYsrp6PULTRX9lY3Xbv0YXqh1Esp9EkV5UfjZWZZwNnR8TZmfmRuutZifVbp35k5rMR0ZQ3aICHgPkU11u1LuHyLHByLRW10Sk7Zi3+L4pW9UexIuiuU2M9PX2cIsTcHxFdjU62Ab5Gw9YGjA5pUw+Mysx5Ed1WI2nSAZcu51E0afgOxdnvI4FGnEnuYWyTp0NHxDuBb/U2bbKcir5VZt48tJVVHAC8CxhL0ZSn6z/oMxQHXWtX/t0Mp7ieru1UvwYa1mPa6RM0p5ld17WzC5t2KUk7EfE+ijOKW1CE7fdm5sJ6q2rrOxQdkP+DluXv6mBYXAUddFaky7KI2C4z/xsgIiZQ83+4nuqYe70GrouIv+g5mJk/qaOYXvwxIqZk5q8BImIqxdHyRijr+nVEfKtrOnQUC4uPy8yn6q2uvYh4I8V1yss7TTZtyjnFWpB/T9FC/94oFkf+Rs01tdoiM0+LiH+iWHMP4O7M/FOdRbWTxcLHv46IbbLBbeqBx8uA0NW981Dg4b4fUo/MvDsihmfmMuCiKNbXbZrrIuINmXlD3YX0YjPg9igWZr8NeIziNWki8JcUXSdPq6+8QvmefklEvDUzv1t3Pb3JzGURsVVErNOkS3P6MCcirqf7eqXX1lhPq4Mi4uM086RJO9tQLEGxoO5CVuLFzDx/5ZsNPq9ZXAURMQWYCnyG4ih5UoSvR4AfN21nNyJeT7HI8D0UR/deDbw7M9t1KatFRPwNRTfMV1McvGjctUFdyu5+XdYD9gBua1h3tz0p3kxaz968PTPn1VdVVUT8mOLs4giKhZAfA27K5nVt/XeKKTb7URzdO5SisUCfDYXq1BK+G9HFESCKBc83objeZg5wc8/rF5skIn7Iig6ejWxTXx78u4BiivlTFMvmvDMz76uzrp4i4ifAX1H8/fwPRaB9VzZk7bUu5RTub1Ccrfk/GvheVJ4N2x/Yh2IphT9RdJW+rmkHNiLi0xQdrp8ub28CfCgzP15vZSuUr+9TKZbMaP07P7fXB9WovJZ+X4r/mz/JzO/VXBIAEfFZimVdNgBa+0808W+oU2aOEBFnUEyVrX35O8PiKiibMZxFcc3ffRR/COMoAtlHm3h0KiLWZUV3t99luch0U0TE3cBbKBZp7qj/jBExjuLN8O1119Kq/DffkeLf/K7MXFpzSRVRLnwdEe+hCDanN7ErZldNLZ9HA1dm5hvqrq1VJ4TvsvnS6yjWiNsHeIAiOM5p4I5u49vUdymv/RzW4M6Dr6Y4oDqSYqr5RsC/dc14aYqy2+ib6cD3oibqeo3vMdaIBjddIuLMduOZ2aip8V3K60H3pLim9tbM/J+aSwKKfY4sOnHPzsyD665nZSLim8BHmva+01M0aPk7p6Gumn8GRgOv7npjjogNKVpYf44VC37WKsp2/206Om4XEU3r5PggcGeHvjkvpliTq2m2BSZQnP3csfw3/1bNNfU0IiK2omhf/bG6i+lD1zTJP0bEGIrrRLatsZ7ebJSZz5Th+6Ku8F13Ua2y6Mg7p/ygnCo7AzgvIl6VmXvUWV+rzLypDDnbZ+YPyut+h9ddV5fyDNMmmfl4Zj4XESMj4r3ABzNzx7rra5WZXbMcnge6msRdTjGNrkkW0bnvRU00vCtEAETE+sC6NdfUTVNDYTvla/sngB9SHAj+UkR8MjMvrLcyAH4B7EpxXWon2Aq4KyIaO3MEIDMbs69hWFw1fwO8pvXNpNxBex/wOxoSFimuX/gh7Ts6Nq2T4ynAtRFxEw1fszK6L5A6jGL6SqMaBpXXDbyBYiH56ymaDdwMNC0sfpKivpsz89ZySt2immtq5/sRsTFFy/JfUfz7/0e9JbXVEeE7Ij6TmacCZOa9wJcjYjzF9N7GKIPXscCmFM0btqZoNPD6OusCiIjDKTqgPhcRi4AzgK8DtwJ/V2Npq2Lvugto42Hgx+WU6Ua/F3WIbwA3RrHkRwJHA43pURARfwf8A8V7JRTTec9t4IHVLh8GpmXmEwARsRnwc6AJYXFkFGsCvrbNSYqmnaCA8qBVJ4iInYBJdO+ZcOmQ1+FBtP6LiP/KzNes6n3qXUTcAPwvPdqVZ7lMSZNE9wVSXwTuy8yf1VVPOxFxB0WI/VVmTikDxFeadsSsE5XTe9fLzD/UXUtPUSzd848U4fv9Zfj+bGa+tebSumk3Da2h048XUFyTfEvXVLqIuCMzd663MoiIO4E3l01jdqU4qn94U65f6o+IeCAzt6m7jlbRyzrKTXwv6hRRNAV8PcWZsBsy8/qaSwKWd5Y9BfgQxUHAoDgz9s8Ur5vfrLG8tiLiRmBG12UlETESuDYbsDZgROxLcaDqbRRr/LbKbNgaoLB8eny3mSNNm8pfvia9jiIsXksxE+fmzBzyg6ueWVw1CyPiyJ6pvnzh+V1NNfUqIk6kuJ7yWYo1uXYFTmtYt7dNm3b9Vx+uAJ7PoqMfETE8IkZl5h9X8rih9Keyy9uLEfEKioYSQz6/vTcRcUpm/nOPs7TLZYPW1+wSEa+lZZ2jclrvkB/Z60sW7cq/03L7HoqFhhuhnH3xfmBCj+mxr6A4Ot40L2Tm0iiXpYiIEbT5/1qTpZl5N0Bm/ioi7m1iUCyDbNu7aNayLkC3dZQ3yHIt0CaKiFcCnwbGZOaMiJgE7J2ZX6u5tIrMvA64ru462jgeOKTHdbM3lD0UvgU0JizGioXulwC3RMRsiteigykacNUui+Vabo6I+U38f9hTk2eO9HAoMAW4PTPfXf7t1zKzybC4ao4HroyIoylaVydFx7z1gUPqLKwXR2fmv0bEARQLjr6bIjw2KSz+IJrdrrzVjRRd/f63vL0+xe+yEQvel24vp01eSLGe4TMUR06b4rfl5/m1VtFPEfF1ijeTBaxYdiaBRoTFDgrf36LYaTyb7u39n62js1s/3BQRH6VYWPyvKYLuNSt5zFDZsmUHEmB06+0GTZv8fB/3NfHg6t4U636OBraJovv5cZn5/norq7iY4n28a7r5f1EsMt+onfSIeJYVr0kjKQ4QPNeQzpgbtWuwlJn3RMRGdRTUh66F7v+7/Ogyu4ZaehURWwKvjogrKP7dF1I0snq070fW4njKmSMAmbmorL9p/pTFUk4vlv1RHqWmg/+GxVWQmUuAPSNif4p114KiZfWN9VbWq67FcA+iaHrx64juKzg3wPHAKRHxAg1tV95ivczsCopk5v9Gsxa8JzOPK7/8tyjWZNowMxsTFjPzmvLz8mtXomhlPTozm3hx/HRgUoObXnRE+C6n7v4BeHvZnOWVFO8/oyNidAO70p0GHEMxPf44iilATblW9aus2IFsd7sRMnO/umtYRV+kuMb7aijWhI026+o2wOaZ+e2I+AhAZr4YEY1aPxkgM7v9n4yIN1PsoP//9u493tax3P/457vImaQkVAgR5ayEDjrYtbOlnNK5/JTqV0qnXbtd0WmXdJIoJJUKRUo7tOVQOmA5rEXolyiVXYmQEmv5/v6477HmmMd1mmvezxi+79drvuZ4njHnmtdrrjnG89z3fd3X1QVT9XftUqbQQKRBS9qFMiH4JcpEai+t9xJJL+nadh26nTnS77I6+X8cZYHqbzRaTc5gcQnY/iGlgEzXza57AjcC3lXTEu9fyPfMqLEXlI67W9J2vcGXpO2Z+qLTRC2AsbHtD0l6lKTtbc9uHVc/SV+jNJGfT3kTfLCkT9g+om1k41wNPILuNjsfN/juMkn/l1KQ5Y+MvBcZ6NSeRdv3Uy7Qx7WOZaxBuHkcVLZvHjOf2rlBGOU69FDqza2knSgTMZ1m+9uS/n3hXzkjHidpoklUAZ2sPSFpbco+yy0ZXeykC32ej6Tso76i79yZks6gFON6UpuwJtXlzJEF+rIajpV0NmXyv0mV8wwWh9uBlGInv7b9d0lrUVJRO0WlWe+mjH4DvKhdRJN6M3CapD/U43XpWPl3SZ+lpPs8ldIT9G5KLv6OLeOawBa1kvBLKKs276QMGrs2WHwYZa/yJYxUSLQ71ktK0g6UtLQN6Htf71rhGMpraLNeRb+uqQWiJp1h7uDvM6bPzXV/smvxkDcxsnLfJYdSVj83lnQxsDYdqyYMCxrI98yiZGl0ZfWmeaGqJXAyJd14D8pE6yso/XS7YI0xA0UAbF9ZFym6psuZI6NIWp++67qkp7a4P85gcbg9GbjSpQ/XSylpAZ9uHNMoKr2DDgEeSdkXthOlul8XZstGqS0eNgc2o8xAXmf7vsZhjbWz7e0kXQFg+7Z649M1D5L0IEoT7M/avk9SV24k+r2/77GAXYED2oQypZMppdVHVRXuoJvp9irICykpsjePOb8B8IfxXx5D5GDK9XF9Sg/dcynbJDqlFjV6GiPXoes7eB2C0a275gE3UYqyNDfRfsUB8FDbJ0g6xFzvb8oAACAASURBVPaFlNWxC1sHVUnSQ2zfPubkWpSJgk7pZY5IOomyUvv7Lm41kfRRyoLELxhdMyGDxZhWxwBb143676BsgP8ypQ9jVxxCWfX6me3d6mCsk2lWdX/iocAGtg+StKmkzWyf1Tq2PvfVPYC9FKWH0s3Bw+cpNw9XAReplLHu3J5Fl+bs2wAvppQFv5GyUts1f7Y9tmR5F/2a0svue3Szl90ngXd7pJE8sCAF7JNM3Ls2JjBFNVSgDHpmKpZFYftWBqBPpUqbnLNtX6PSV3c7SR/s0u+z7kueY/uTrWMZIr0JgVskPY8yefXIhvH0+ySlmuzbGCmotz3w0fpcJ0g6FjiqvnYeTFmYmA+sJelttr/eNsJx9qJk4vxzoV+5jGWwONzm2bak5wOfrrNSr1jod82se2zfIwlJK9q+TtJmrYOaxImUVMleQ+nfUdoVNB8sSlre9jzgaOBbwNqSDqMMcDo3+Lb9GeAzfad+I6kzBTEkPRZ4EWUV8S+U9B91uGjH+yQdT6nY2z8I61oz5N/WjxXqR9dsONGeENuXSdpw5sNZNJLOsr1H6zjG6FVDXYmSgngVZSVsK0oVwl0bxTUhSZ+Z4PQdwGW2u1R58j9tn6bS2+5fgI9TJoY7sy/MpX3TnnRooDAEPlgHOG8FjgLWoKT1N2f7C3V7zgcoK3W9aqgf7O2r74in2D64Pn4V8Evbe0l6BKVad9cGi7+mbCvKYDGWqbtqxbSXAU+ps31d62/1u1rt6dvADyTdTnfTvTa2vb+kAwBs/6ND1WUvAbaz/WVJsyktPgTsa/vqtqGNp9E9QI8HtqXsI+hKC5XrgB8B/+ba007SW9qGNKVXAZtTXt/9hWM6NVgcgOIsK03x3MozFsXiW791AGP1JlYkfQN4je259fjxwNtaxjaJlSivoV6/0r2Ba4ADJe1muxM35oykoz0POMb2mZLe3zCeyfyk7qE/hbJ3HujWirKkbWxfOebcc136Q3ZKXwbTHUDvtdWVv8lefM0nzhfi3r7Hz6a+1m3/b3du5UAjrbD+Dlwpaewk8Iy3xMpgcbjtT0mfe3V9MTyajhUQsd3rT/l+SecDDwbObhjSVO6VtDIjKZ4b04EZn2rBO53tayg3OV3W3wN0bbrXA3Rvysri+bUK2Tfo+x130Na2O1+0oeMV/QAulXSQ7VFVUCUdSMkq6KpxxSU6ZPPeQBHA9tU1tbtrNgGeUTM0kHQM5f3o2ZS9wF3xe0mfp0wIflTSinRwXxgj/YcP7ztnulWP4IuSXmr7F7AgxfcdlFWmQXAopeVLLJq/StoD+D2wC6XITa91RpcmA3utsGZTW/m0lsHiEKsDxG9RKo0C3Aqc0TCkUereujm2Hw9lf1jjkBbmfZSB7KMknUx5s3ll04hGrK3RjbpH6dCesJ5O9wC1fQZwhqRVKfsG3gKsU28gz7DdlUFtz88kbdG76emwLlf0g5LWdUat0tsbHO5ASZl9waTf1ZjtV7eOYQrX1hTpr1IGCy+lm1VG1wdWZaQA06rAejWlsiuTglC2FjwH+Ljtv0pal1LcqmsOtP3r/hOSmjQUn8J+wKkq7aZ2pQwedm8b0mLpzDVzQLyWsv3lEcCbbf9vPf9M4HvNohqj1wqr3n/cY3t+PV4OWLFFTOpgAaCYJpIOAl4DrGV7Y0mbAsfafmbj0Baog653uXtNuSdUC8bsRHmT/lktitCcpFso+1YmvHh0Lf1P0omUm7ONgK2B5YALbG/fNLAp1Mpu+wL7d2glDABJ1wIbUwrw/JPyd+CutXqQNNv29pLm9GKTdKHtLhXdou6ffXw9vMalt24sAUkrAa+jtPOBUsnvGNv3tItqvLp6/B7gAsrr56nAhyn7mN5vu/mAbOwEa5dJutz2dmPOze7ae3wtqnc6ZbXp+bb/3jikRSbpt7Yf3TqOWDYk/Qx4lu2/1ePVgHNt7zz1dy6DWDJYHF6SrgSeCPzc9rb13NwupatJ+iGlGuoljN7XsGezoCYh6UDbJ/QdLwe8pwsDsYkuzF1Wb3p6PUD/Wgdi6/enq8Wiq9Vkxxlb1bM1ST+zvZOkcygzvH8Avml748ahxTJU0/cfbfv61rFMpa7SPZEyWLzEduf2z3d9grUOvrYEPsboFc81gLfb3rJJYH1UWkv13/w+AvgrcA9Al66lku5i4v6UAla23ZkMwZoSvTewIaP7/R4+2ffE5CRdaXubhZ2bCZ35I4tl4p+27+1l99W87K7NDjQfaC2GZ0ram5Kq8lDKHruupM4OVDpK7XN0OSzY+3kAZY9g52fMu8j2byQ9BHgUo9/XOzVYpMMV/QZZnXxZzXbn2s/UqphHUFJ5N6r7FQ/v4oQgZbBwC2U/7SaSNnGDBtgLsS5wjaSuTrBuRkkzX5PRrWbuAg5qEtF4+7QOYFHZ7mJT+8mcSUnjnk136jkMsrslbdcrCiVpe+AfLQLJYHG4XSjp3cDKkp4NvB7oUhnjQdinuIDtF0van1Ls4O/AAbYvbhxWT2dSixdFncHvFWDaCvgI3Wx2PxAkfYCyf/YGRiaEulZMAuB223cwuqLfLm1DGkySvkbZ9zmfcnP2YEmfsN2pImaUvd5PpKR3YvtKdbANiaT/Q+n7+0jgSsp2g5/SvddQpydYa5uRMyU92fZPW8czEds3AEjaEbi2L81vdcpgN5bMI20/p3UQQ+TNwGm1LQmUiaL9WwSSNNQhVmebexu2BZwDHO8O/adPkmJxB6Ua1FvHbpBvqe75PIkyWHwcpY/QoYO0x6G1uo/2AMoN2an140zbGzUNbMBJuh54gu17F/rFDU2yj2mgUqi7opeOVIvxbA+8E5jdwX2qP7f9JElX9G2HmNPBOOdStkT8rP5eNwcOs93k5mzQSfoY8EHKSsjZlL3pb7b91aaB9anpqNvXTJfePdOlXdtXOSgkfYHS9L6T20mmKgIInSwEiKQHUSYwBFxn+74WcWRlcQhJOq8WsfmI7XcCxy3sexr6BGXf0tcoL4YXUfYPXA98EXh6s8jG+y7wBtvn1cqdhwKXUvZnxKI5mjJb/2LblwFI6szkxQC7mpL29afWgUxE0pMppfTHVu1dg1LcKBbfg+qNxF7AZ23f19HX0tWSXgwsVyfc3gT8pHFME7nH9j2SkLSi7eskdW6VacwE6wqU3qp3216jXVQT2t32OyS9APgdpTjY+ZSquF0xqzdQhLI9or6mYsnsCrxSUlcLrQ1SSi+SVqHcZ25g+yBJm0razCM9N2dMBovDaV1JTwP2VGmIPGo/mzvUFBd4ju0n9R1/oRbBOLym0HbJE3t7gurq7JGSOtEDZ4CsR7lp+ISkdSgri7k4L72PAFdIuprRzXu7so9pBWA1yjWn/4J9JwO0f6hjPg/cBFwFXFSLHHVuzyLwRuA/KH+XX6dkuHygaUQT+52kNYFvAz+QdDtlIrNTxu5hk7QXJc23a3rv6/8KfN32bepOd6SeGyW9DvgCZQD+OsprKpbMc1sHMJUuFCNcTCdSthg8uR7/DjgNmPHBYtJQh5CkfSjpp7sy0tyzx10q+y/pp8AngW/WU/tQUjt3alX1aSxJ77D9sfp4X9un9T33YdtdG9QOBEmPpKwkHwCsQulfmN/lEpB0DWXwMBfonynv1J5gSRt0rULrMJG0vGtT+VhydbL1wcDZXU/thpEqw63j6Cfpvyir3v+gDGbXBM4aMzncVJ2wPJqSwWTKyucbbf+xZVyDTNLWwFPq4Y9sX9Uynn69ezlJRzFBsUfbb2oQ1qQkXWZ7hzEp/FfZ3nrGY8lgcXhJ+k/bXZzBXUClSe+nGZk5+SmlAfrvKXsJftwqtp7+PVVj91dlv9X0qOleLxrAmb9OUAd7FU5E0mOBtzG+tHpnJrAGRb3R/TClcfxzJW0BPLm/vU9Lkr7LFNW3O7TqvYBKO6R1GP232akWFZJe2Hc4C9gBeJrtJ0/yLc3UCs132p5fU+rW8Egj9Bgykg6hVLw9vZ56AfAF20e1i2qEpH+z/V1Jr5joedsnzXRMU5H0E0rxwottb1crx3/d9oxnEmSwOORq2fJeM+QLWuQ6D7oxszoLHk90HNGCpE9Q0vy+w+g01C6lnCPpKuBYSmrN/N5527ObBTWgJH2fkqb0H7a3VmmNdIU70ke3rs5NqoOr3m+kVG79IyOr813abwWApBP7DudR0iaPs925/cqSdmb8xNCXmwU0hqTjmHiF6TUNwhl4kuZQJqzurserAj/t4GtoW9tXtI5jYWoXg/cAWwDnArsAr7R9wUzHkj2LQ0zSRyjpHyfXU4dI2sX2uxqGNUpNRTyK8iIw8GPgENu/axrYaJ7k8UTHES30Jiz6U9G62Dpjnu1jWgcxJB5m+1RJ7wKwPU/S/IV900zp2mBwERwCbGb7L60DmYrtV7WOYVFI+gqwMaUNSe/v0kBnBovA//Q9XomyEnZzo1iGgeibBKyPO7dRlVIzYV3K/r9v2L6mdUATsf0DSZdTruui3Bvf2iKWDBaH2/OAbfrKQp8EXAF0ZrBImRn/GqXoCcBL67lnN4tovK0l3Ul5sa5cH1OPV2oXVkRhe7fWMSyi70p6PXAGo1dAb2sX0sC6W9JDqRNWknaitB3qhNqKYqo01E6tNlAGCZ35/Y0l6b1TPO0ObjnZAdiiS626xrJ9Sv9xHeD+oFE4w+BE4OeSzqjHewGdSIvvZ3s3SY8A9qMUVVwDOMX2BxuHBoCksVubbqmfHy3p0S0yhpKGOsRqSsDTezdiktaipKJ25iI9URGbrhS2iWWnth55CfCYWvn20cAjbF/SOLSBJOnBlBS6Xsr5hcDhtjt181tLqo9l24+Z8WAGXL2hOAp4PKV1ytrAPrbnNA2sqtVZJ9W1QkeSTqD0M/seoycyOtF7TdJbJzi9KqWY3UNtrzbDIU1J0mnAm2zfstAv7oi6J+wc25u0jmVQ1felXSmT6Rd1Pd1T0hOAdwD7216hdTwAku4HrgH+3DvV93STIpVZWRxuvXL651P+2J5Kt1YVAW6V9FJKSXUolTE7nQYU0+JzlH1BzwAOB+4CvkVpih2L74uUAcN+9fhllFneF076HQ3Y3qh1DMPC9uV1X2CvYfP1btSweSJdGwwugt/WjxXqR6fYPrL3WNLqlLTZVwHfAI6c7PsaehjwC0mX0M12PtT2KL0Vk1nAbcC/t4toMElaw/addUHiJvraj0haq2uZI5IeB+xPqb7/F8praKLJmFbeCuxNqST8DUql+L+1DCgri0Oqrtw8krIBfkfKzcTPu1aJrK4ofZZSDdWUZs2HDOCNRiyGXhXZLpSEHgaDskIv6eUTne9S0YuuG1MNcxzbp0/1/EyR9GPbu2p0E3kYadTdtSbynVdvxg+lZGWcBHza9u1to5rYZAWOurKXtd4jPYpSeR3g/i6nzHaZpLNs71EzRyZ6rXcqc0TSzygLFKfZ7lwv1R5JG1EWUJ4P/Ab4sO0rW8SSlcUhZduSvm17e0qFxM6pZcr37tJMY8yY++r/f2+/1dr09QeMxfYPSbv2Ws1I2oUyK9k1/SvHK1HKgl9Ot4pedN2/TfGcGSlb35TtXevn1Rf2tV1Q34PeAWxJ3170rrR1kXQEJVPgC8ATWq80LIztC2t7l95r/pIuVWyt90hn1HukWAq296ifByJzxB3rSToZ2zdKOhNYmZIt9FhKwagZl5XFISbpaOBLti9tHctkJF1g++mt44iZJekllDSQ7Sgz5PsA77F9WtPABpRKI+QvUxqJA9xOKbHdmYbIE6l7Lb+SCaPhJelAj+n9KOm/bHcq3U/SucAplD6gBwOvAP5s+51NA6vqPqZ/UrKFOr9SK2k/4AjgAkqMTwHebvubLePqJ+kYStuRTrUYGlSSzrP9zIWda0XSqbb3m6D4Vu811Il6Hir9x19EWVG8mZKKepbte5rFlMHi8JL0C8p+lpuAu+nYCwJA0ocoN7inUGIEutcfLqafpM0pK0sCzrN9beOQBl6t6obtOxf2tV0g6UHAHNuPax3LIJL0PMavhB3eLqLxaj/Ir9o+uR5/DljJ9qvbRjaapNm2t5c0p3eNlHSh7Sn7RcbEak/VZ/dWE+vK7f90YauBpOVrq5m5wOOAGxh9jzS2GmVMQdJKwCrA+cDTGSnIsgbw/a68v0ta1/YtkxXf6sr2pzoxNAc4E7iTMVWlWxTdShrqcHtu6wAWwc71c/8NThf7w8U0kvRpSqnqo1vHMsgkHQrc0Vu56Q0Sa4Px5Wx/qmV8Y0n6LiMXvuUoN2qntotocEk6lnKDthtwPGV1vovVhF8IfKfeAD0XuM326xvHNJFecaBb6iD8D5R9/7FkZo1JO/0LpYhMF1xCyWrZq3UgQ+K1wJuB9YDZjAwW7wS6dI1ftfYav7j/pKSnUF7vXXE4I9fJTlQ5zsriEKqzPAcDmwBzgRNsz2sbVcQISa+gpKE+ltJz7xTbl7WNavBIuhrYzva9Y86vCFzapSwCGFf0Yh7wG9u/axXPIOutgPV9Xg043fburWODBcVYelYHvg1cDLwXutdbU9IewI8oRU+OoqyKHGa7k3v+u67usdyKkUrn+1OyCJqn9fYXVovpI+mNto9qHcdkJJ0FvHtseyFJOwDvsz3VfvAHtAwWh5CkUyizpD+izOT+xvYhbaMaTdKTKBv1N6YMaF+dNMQHnnpDuTclP//RtjdtHNJAkTTX9hMW97mWulz0YpBI+rntJ9XKfi+krNxc3ZXXUF9lRPV97ulchcSYHpI2AdaxfXGt3NvruXc7cLLtG5oGCEj6HTBpKl+LNL9hIOkNlP/jv9bjhwAH2P5c28gKSVfbfvwkz3XyetkVSUMdTlv0/uhro+EupiYdTSkkcBGwJ/Ap4F+aRhQtbAJsDmwI/KJtKINJ0jq2/zj2XKt4pjJB0YujJHWq6MUAOUvSmpTf5+WUAdlxbUMaMSiVESV9Zqrnbb9ppmIZEp8C3g0L2ricDgtWbz7F1NV8Z8pylPQ+LewLY7Ec1L+1xPbtkg6i9FXugpWmeG7lGYtiAGWwOJwWNGaum7hbxjKZWbZ/UB+fJuldTaOJGSXpo5TVkBsoe9Y+0JuNjMVyBPA9SW+lDBgAtgc+Bny8WVST+w9gx7FFL4AMFheT7Q/Uh9+q6VUr2b6jZUz9JO0I3Oza27f22Nyb0i/s/R1KQz0YuJryPvQHMoBYWhuOTfMDsH2ZpA1nPpwJ3dK1QlBDYpYk9fpV1vZYKzSOqd+lkg6yPWpSTdKBlL2WMYkMFofT1pJ61RAFrFyPu1Rie80xzaVHHbsjjaVjmbkReLLtW1sHMshsf1nSnykb4h9PWV26hrL/4vtNg5tYl4teDISpBmGSujQI+zzwLABJTwX+C3gjsA1lC8I+7UIbZV1gX8qeunmUytzfckeb3Q+AQVi9yYTAsnEOcGotvmXKRMzZbUMa5c3AGbV1V29wuANlQPuCZlEtAkln9fpZNvn52bMYLUg6cYqn3bWy6jE9JG1u+zpJE5YmT8uU4TZJ0Yu5tt/RLqrBIuly4Fm2b6uDsG8wMgh7nO1ODMIkXdVrk1B7/v7Z9vvr8ZW2t2kZ30QkrQ8cABwKvNP2VxqHNHAkfR344SSrN7vb3r9NZKNiWatDkypDQ9IsSmXUXkusc4Hjbc9vGtgYknajTK4CXGP7hy3jWRStizJlsBgRM0bSF2y/RtL5Ezxt22mZMuTGFL24yPYZjUMaKIMyCKuVerepWyGuA15j+6Lec5MVmmilTmAdADybsupwpO3so15Mdb/0GcC9TLB601sRj+EkaWVKsbrrW8cyTCR9seUiSgaLETHjJK1k+56FnYvhImkjyn6he+rxypTKiTc1DWyADMogTNJ/AP8K3Ao8mtLixbVa5km2d2kaYCXpMGAP4FrKKu3ZaTW19AZx9SaWjqQ9KfvoV7C9kaRtgMNt79k4tFhKGSxGxIyTdLnt7RZ2LoaLpMuAnXt9ISWtAFxse8epvzN6BmUQBiBpJ8qewHNt313PPRZYrSsp55LuB34N/KOe6t0U9fb4d6pXaURXSZoNPAO4oJcy2esD2zayWFopcBPN1Pz2nWz/pHUsMTMkPQJYn1J0aVtGCg2sAazSLLAh0noj/EIs3xsoAti+tw4YYxHZ/pCk8xgZhPUGN7Moexc7w/bPJjj3yxaxTGEgWnxEDIB5tu/oaAX+WAoZLEYztu+XdCTw5NaxxIz5F+CVwCMZ3RT5Lmpvrlhq67cOYAp/lrSn7e8ASHo+ZYUsFsOADMIGgu3ftI4hYkhcLenFwHKSNgXeBGQxYAnUtiP/ZfvtrWOBpKFGY3W/yBzgdOeP8QFD0t62v9U6jmHUeiP8VCRtDJwMrFdP/Q54ue1ftYsqIiKWlqRVKL10d6+nzgE+mFoES0bSD4FnduHeOIPFaErSXcCqwHzKnpEu9YKMZUjS84At6evLlUbJDwySVqNcf+5qHUtERCydrq2EDYOaebcpcBpwd+98iz7kSUONpmyv3jqGmHm1ae8qwG7A8ZQG3Zc0DSqWOUkfBj5m+6/1+CHAW22/p21kERGxpGzPl7R96ziGzFrAXyhFg3oMzPhgMSuL0ZTKTuiXABvZ/oCkRwHr2s7AYYj1KqT1fV6Nkoq8+0K/OQbWRI2FUwU3WpI0l5EKqKOeItVQIxZZl1bCYnplZTFa+xxwP2Xm5APA34CjgZTSH269PQx/l7QeZfYsVQmnQa0yvJrtO1vHMoHlJK1o+5+woM/iio1jige2rlYOjhg0nVkJGwa1zdAxlF7Ej5e0FbCn7Q/OdCwZLEZrT7K9naQrAGzfnlL6DwjflbQmpYHv5ZQLynFtQxpckr4GHEzZ+zsbeLCkT9g+om1k43wVOE/SifX4VcBJDeOJB7hUQ41YenXP4hzbn2wdyxA5Dng78HkA23PqtX7GB4uzZvoHRoxxX32TMYCktSkrjTGk6srXebb/WiuibgBsbvu9jUMbZFvUlcS9gP+mNGt/WduQxrP9McqF7nHAFsDZlP//iCYk/bh+vkvSnX0fd0nq4up8ROfYng/s2TqOIbPKBFuy5rUIJCuL0dpngDOAh0v6EKXQyX+2DSmWpbH9NWtK4j/bRjXwHiTpQZTB4mdt3yepqxvS/5cyIbQfcCOQFirRjO1d6+cUW4tYOj+R9FngFEbvWby8XUgD7dbabqq3mLIPcEuLQDJYjKZsnyxpNvBMSkGBvWxf2zisWPbOlbQ36a85XT4P3ARcBVwkaQOgM6side/Fi4ADKHtaTqEUWNutaWARY0h6OKPb+fy2YTgRg2Tn+rm/BZYZvYcxFt0bgC8Am0v6PWVy9aUtAkk11GhK0ldsv2xh52K49PXXnEcpdpP+mtNM0vK2m6SsjCXpfuBHwIG2f1XP/dr2Y9pGFlFI2hM4ElgP+BMlPfpa21s2DSwiHtAkrQrMatmXOHsWo7VRF+K6fzG9eoac7dVtz7K9gu016nEGiktI0jqSTpD0/Xq8BfCKxmH125uSfnq+pOMk9TIJIrriA8BOwC9tb0TJdrm4bUgRg0PSgyV9QtJl9eNISQ9uHdeg6l3XgW/avkvSFpIObBFLBovRhKR31dWlrfqKCdxFmdE9s3F4sYxJOm9RzsUi+xJwDmVVBOCXwJubRTOG7TNs7w9sDlwAvAVYR9IxktJbM7rgPtt/AWZJmmX7fGCb1kFFDJAvAndR9qPvR9kKceKU3xFT+RIdua5nsBhN2P5ILShwRN/K0uq2H2r7Xa3ji2VD0kqS1gIeJukhktaqHxsy8oYYi+9htk+lVhKu6afz24Y0nu27bZ9sew/gkcCVwL83DisC4K+SVgMuAk6W9GkaVR6MGFAb236f7V/Xj8OAbDVYcp25rqfATbT2fUlPHXvS9kUtgoll7rWUmbH1KP0Ae6mIdwJHtwpqCNwt6aGMVE3bCbijbUhTs30bpTDP51vHEgE8H/gHZdX7JcCDgcOaRhQxWP4haVfbvXY0u1BeU7FkOnNdT4GbaErSd/sOVwKeCMy2nepZQ0zSG20f1TqOYSFpO+Ao4PHA1cDawD625zQNLGJASPqo7Xcu7FxETEzS1sCXKRMtALcDr7R9VbuoBleXrusZLEanSHoU8DHbB7SOJZYdSfsCZ9dN2+8BtgM+mH5MS07S8sBmlNXa623f1zikiIEh6XLb2405N8f2Vq1iihhEktYAsN2Z9k2DqivX9QwWo1MkCZhj+wmtY4llp3cTJmlX4CPAx4F3235S49AGiqQXTvW87dNnKpaIQSTpdcDrKXurbuh7anXgYttN+ppFDApJhwJ32D5hzPk3AsvZ/lSbyAafpJ2BDenbNmj7yzMeRwaL0ZKko6j52JSCS9sAN+UCPdwkXWF7W0kfAeba/lrvXOvYBomkqSrN2farZyyYiAFUS/s/hDJp1V9s6a66rzYipiDpamA72/eOOb8icGlW55eMpK8AG1MKwfUK29j2m2Y8lgwWoyVJ/b3g5lEGiultNeQknQX8HngWpa/mP4BLbG/dNLCIeMCqmQ6b2j5R0sOA1W3f2DquiC6TNHeybLCpnoupSboW2MIdGKilGmo0ZfskSStQ+q8ZuL5xSDEz9gOeA3zc9l8lrQu8vXFMA03S84AtKYWiALB9eLuIIgaHpPcBO1D2B50IrAB8FdilZVwRg0DSOrb/OPZcq3iGxNXAI4BbWgeSwWI0JelfKaXzb6Bs4N1I0mttf79tZLEsSFqjbnpfidKcndp38Z/AZQ1DG2iSjgVWAXYDjgf2AS5pGlTEYHkBsC1wOYDtP0havW1IEQPhCOB7kt5Kff1QMoY+RqlHEEvmYcAvJF1CuUcCwPaeMx1IBovR2ieA3Wz/CkDSxsD3gAwWh9PXgD0oPRbNSJ9F6nEa+C6ZnWvBoDm2D5N0JJDiNhGL7l7bltTrabZq64AiBoHtL0v6M3A4pc2DgWuA92Xif6m8v3UAPRksRmt/6g0Uq18Df2oVTCxbFN+IvgAADiJJREFUtveonzdqHcuQ6TU+/ruk9YC/APkdRyy6UyV9HlhT0kHAqymr9BExBUkHAOfaflrrWIaJ7Qtbx9CTwWK0do2k/wZOpcxG7Qtc2msJkNL/w0vS+sAGjC4JfVG7iAbaWZLWpKQDXU55LR3XNqSIwWH745KeDdxJ2bf4Xts/aBxWxCDYADhN0oOA8yiZYZd0oTDLIJL0Y9u7SrqLkW4BUDKxbHuNGY8p/5fRUkr/PzBJ+iiwP/ALRpeEnvFc/GFTy5WvZPuO1rFEDCpJywEvsn1y61giBkHd4/ssSvG6JwLXAmcD54wtfhODJYPFiJhxkq4HtrL9z4V+cUxK0o7Azbb/tx6/HNgb+A3w/vSJi5iapDWANwDrA98BflCP3w5cafv5DcOLGBiSHmX75r7jLYDnArvb/pd2kQ2mWvxvrLts3zfjsWSwGC1J2gh4I7Aho9MRs8I0xCR9H9jX9t9axzLIJF0OPMv2bZKeCnyD8nraBnic7X2aBhjRcZLOBG4Hfgo8E3gIpW3GIbavbBlbxCCRNNv29q3jGBaSbgIeRXl/ErAmpY3Gn4CDbM+eqViyZzFa+zZwAvBd4P7GscTM+TtwpaTzGF0S+k3tQhpIy/WtHu4PfMH2t4BvScqNbsTCPabXNFzS8cCtwKNt39U2rIiB8zNJO9q+tHUgQ+Js4Azb5wBI2p2S4nsq8DngSTMVSAaL0do9tj/TOoiYcd+pH7F0lpO0vO15lFWR1/Q9l/f3iIVbkNJle76kGzNQjFgiuwEH1xWxuxkpyLJV06gG1w62D+4d2D5X0odtH1prE8yY3ExEa5+W9D7gXEavMF0++bfEoLN9UusYhsTXgQsl3Uppn/EjAEmbAClwE7FwW0u6sz4WsHI9blZ5MGJAPbd1AEPmNknvpGwvgZI9dHstvjWjmXjZsxhNSfoI8DLgBkb++G37Ge2iimVF0qm295M0l9EloQHIDOTik7QTsC6lz9Xd9dxjgdUy6RIRETNF0q7AprZPlLQ25Tp0Y+u4BpGkhwHvA3alTF79GDiMMhH86DE9ypdtLBksRkuSrqNUxby3dSyx7Ela1/YtkjaY6Hnbv5npmCIiImLp1CyxHYDNbD9W0nrAabZ3aRxaLKWkoUZrV1EqPP2pdSCx7Nm+pX7+DSwoW5/3oYiIiMH2AmBb4HIA23+ovRdjCdQMobcxvlvAjGfe5SYtWlsHuE7SpYzes5jWGUNM0muBwyn77HrpDQYe0yyoiIiIWFL32rYkA0hatXVAA+404FjgeGB+y0AyWIzW3tc6gGjibcCWtm9tHUhEREQstVMlfR5YU9JBwKuB4xrHNMjm2T6mdRCQPYvRAZLWAXash5fYTkrqkJN0NvBC239vHUtEREQsPUnPBnanFGQ5x/YPGoc0sCS9n7JF6wxGZ97dNtn3LLNYMliMliTtBxwBXEB5c3kK8Hbb32wZVyxbkrYFTgR+zug3wTc1CyoiIiKWiKSNgFts31OPVwbWsX1T08AGlKSJqsja9oxv18lgMZqSdBXw7N5qYi21/D+2t24bWSxLki6hlIGeS1+/oPRfjIiIGDySLgN27lW3l7QCcLHtHaf+zui67FmM1maNSTv9CzCrVTAxY+bZPrR1EBERETEtlu9vg2b73jpgjMUg6R22P1Yf72v7tL7nPmz73TMdU27Ko7WzJZ0j6ZWSXgl8D/h+45hi2Ttf0mskrStprd5H66AiIiJiifxZ0oJK9pKeD6SI3eJ7Ud/jd4157jkzGUhP0lCjOUkvBHal7Fm8yPYZjUOKZaxLufgRERGxdCRtDJwMrEe5n7sZeLntXzUNbMBIusL2tmMfT3Q8U5KGGk1I2oSy8fli26cDp9fzT5W0se0b2kYYy5LtjVrHEBEREdOj3rftJGk1ymLUXa1jGlCe5PFExzMig8Vo5VPARHnXf6/P/dvMhhMzoa4iT6pOHERERMQAkPRS21+VdOiY8wDY/kSTwAbX1pLupKzOrlwfU49XahFQBovRyoa254w9afsySRvOfDgxQ3qTAA8HdgZ+WI93o7RPyWAxIiJicKxSP6/eNIohYXu51jGMlcFitDLV7MjKMxZFzCjbrwKQdBawhe1b6vG6wNEtY4uIiIjFtnH9/Iv+yp0xPFINNVq5VNJBY09KOhCY3SCemFkb9gaK1R+Bx7YKJiIiIpbIv0p6EOMrd8aQyMpitPJm4AxJL2FkcLgDsALwgmZRxUy5QNI5wNcpG7YPAM5vG1JEREQsprMpLTJW7dtfB2WPnW2v0SasmC5pnRFNSdoNeHw9vMb2D6f6+hgekl4APLUepmVKRETEgJG0ou1/SjrT9vNbxxPTL4PFiGhO0q7AAbbf0DqWiIiIWDSSLre9naSv2H5Z63hi+iUNNSKakLQNJf10f+BGUgk1IiJi0Kwg6RXAzhO1x0pLrMGXwWJEzBhJjwVeRBkk/gU4hZLhsFvTwCIiImJJHAy8BFiT8T2yTSaCB17SUCNixki6H/gRcKDtX9Vzv7b9mLaRRURExJKSdKDtE1rHEdMvK4sRMZP2pqwsni/pbOAblIppERERMYAkPRzYQNI3KauJvwCOtv2ntpHFdEifxYiYMbbPsL0/sDlwAfAWYB1Jx0javWlwERERsVgk7QJcShkkfhn4an3qkvpcDLikoUZEU5LWAvYF9rf9jNbxRERExKKR9DPgdbavGHN+G+Dztp/UJrKYLhksRkRERETEYpP0C9tbLO5zMTiShhoREREREUtCkh4ywcm1yDhjKOQ/MSIiIiIilsQngXMlPU3S6vXj6cD363Mx4JKGGhERERERS0TSHsA7gC0ZqYZ6hO3vNg0spkUGixERERERETFO0lAjIiIiIiJinAwWIyIiIiIiYpwMFiMiIiIiImKcDBYjIiIiImKJSVpH0gmSvl+Pt5B0YOu4YullsBgREREREUvjS8A5wHr1+JfAm5tFE9Mmg8WIiIiIiFgaD7N9KnA/gO15wPy2IcV0yGAxIiIiIiKWxt2SHkrps4iknYA72oYU02H51gFERERERMRAOxT4DrCxpIuBtYF92oYU00G2W8cQEREREREDTNLywGaAgOtt39c4pJgGSUONiIiIiIglJukNwGq2r7F9NbCapNe3jiuWXlYWIyIiIiJiiUm60vY2Y85dYXvbVjHF9MjKYkRERERELI1ZktQ7kLQcsELDeGKapMBNREREREQsjXOAUyUdS6mIejBwdtuQYjokDTUiIiIiIpaYpFnAa4FnUgrcnAscbzu9FgdcBosRERERERExTtJQIyIiIiJisUk61fZ+kuZS0k9Hsb1Vg7BiGmVlMSIiIiIiFpukdW3fImmDiZ63/ZuZjimmVwaLERERERERMU7SUCMiIiIiYrFJuovR6aeqxwJse40mgcW0ycpiREREREREjJOVxYiIiIiIWCqStgaeUg8vsj2nZTwxPWa1DiAiIiIiIgaXpEOAk4GH14+TJb2xbVQxHZKGGhERERERS0zSHODJtu+ux6sCP03rjMGXlcWIiIiIiFgaAub3Hc+v52LAZc9iREREREQsjROBn0s6ox7vBZzQMJ6YJklDjYiIiIiIpSJpO2BXyoriRbavaBxSTIMMFiMiIiIiYrFJWgk4GNgEmAucYHte26hiOmWwGBERERERi03SKcB9wI+A5wI32X5z26hiOmWwGBERERERi03SXNtPqI+XBy6xvV3jsGIapRpqREREREQsift6D5J+OpyyshgREREREYtN0nzg7t4hsDLw9/rYttdoFVtMjwwWIyIiIiIiYpykoUZERERERMQ4GSxGRERERETEOBksRkRERERExDgZLEZExAOepEdKOlPS/5N0g6RPS1qhYTx7Sdqi7/hwSc9qFU9ERDwwZbAYEREPaJIEnA582/amwGOB1YAPNQxrL2DBYNH2e23/T8N4IiLiASiDxYiIeKB7BnCP7RMBbM8H3gK8WtKqkj4uaa6kOZLeCCBpR0k/kXSVpEskrS7plZI+2/tHJZ0l6en18d8kHSnpcknnSVq7nj9I0qX13/mWpFUk7QzsCRwh6UpJG0v6kqR96vc8U9IVNaYvSlqxnr9J0mH1Z8yVtPnM/QojImIYZbAYEREPdFsCs/tP2L4T+C3wf4CNgG1tbwWcXNNTTwEOsb018CzgHwv5GasCl9veDrgQeF89f7rtHeu/cy1woO2fAN8B3m57G9s39P4RSSsBXwL2t/0EYHngdX0/59b6M44B3raYv4eIiIhRMliMiIgHOgETNR0W8FTgWNvzAGzfBmwG3GL70nruzt7zU7ifMsAE+Cqwa338eEk/kjQXeAll4DqVzYAbbf+yHp9UY+w5vX6eDWy4kH8rIiJiShksRkTEA901wA79JyStATyKiQeSkw0u5zH6urrSFD+z9/1fAv5vXSU8bCHf0/vZU/ln/TyfsuoYERGxxDJYjIiIB7rzgFUkvRxA0nLAkZSB3LnAwZKWr8+tBVwHrCdpx3pu9fr8TcA2kmZJehTwxL6fMQvYpz5+MfDj+nh14BZJD6KsLPbcVZ8b6zpgQ0mb1OOXUdJaIyIipl0GixER8YBm28ALgH0l/T/gl8A9wLuB4yl7F+dIugp4se17gf2Bo+q5H1BWBC8GbgTmAh8HLu/7MXcDW0qaTSmoc3g9/5/Az+u/cV3f138DeHstZLNxX6z3AK8CTqupq/cDx07X7yIiIqKfyjUyIiIilhVJf7O9Wus4IiIiFkdWFiMiIiIiImKcrCxGRERERETEOFlZjIiIiIiIiHEyWIyIiIiIiIhxMliMiIiIiIiIcTJYjIiIiIiIiHEyWIyIiIiIiIhx/j8sx8VBkqBs8gAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#top 20 occupations\n",
    "order_occupation = df_loans.Occupation.value_counts().index.tolist()[:20]\n",
    "fig = plt.figure(figsize = (15,5))\n",
    "\n",
    "ax_1 = sb.countplot(data = df_loans, x = 'Occupation', color = base, order = order_occupation)\n",
    "plt.xticks(rotation=90)\n",
    "for p in ax_1.patches:\n",
    "    height = p.get_height()\n",
    "    ax_1.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total),\n",
    "            ha=\"center\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Occupation might be the least useful/easy variable to analyse. There are many unique occupations in the dataset (almost 70) and most of the loans were given to \"Other\" and \"Professional\" workers, which doesn't give enough details on the type of profession. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Discuss the distribution(s) of your variable(s) of interest. Were there any unusual points? Did you need to perform any transformations?\n",
    "\n",
    "> Initially, I cleaned the data to remove some unusual points (for instance the rows Prosper Scores above 10). During the univariate analysis, I had to change the data format of some columns to be able to plot it.\n",
    "> Also, in order to find out the duration of each loan I had to create a new variable that takes the closing date and subtracts from it the creation date. This returned the number of days between the two dates, which I then transformed in months for convenience. \n",
    "\n",
    "### Of the features you investigated, were there any unusual distributions? Did you perform any operations on the data to tidy, adjust, or change the form of the data? If so, why did you do this?\n",
    "\n",
    "> The univariate distributions are all generally as expected, with some interesting insights that we will investigate further in the next step. \n",
    "- Term of the loan:\n",
    "    - The more recent years have seen an increasing trend in number of loans being **created** and **closed**.\n",
    "    - Most of the loans have a 3 year **term**.\n",
    "    - The **duration** of the loans follows a bimodal distribution with peaks at the 1 year and 3 years points. \n",
    "- Credit Rating/Score:\n",
    "    - By looking at the Credit Grade, Prosper Rating and Prosper Score, we found out that most of the borrowers have a **credit score** lying in the middle, ie the credit score roughly follows a normal distribution. \n",
    "- Loan features:\n",
    "    - Most of the loans are current or completed, with only a very small portion being with **status** past due, defaulted or charged off. \n",
    "    - Among the **reasons** for which loans have been taken out, the most common are: Debt Consolidation, Home improvement, Business. A quarter of the reasons were classified as \"Other\". \n",
    "- Borrower's features:\n",
    "    - As expected, most of the borrowers (ca 95%) are **employed**.\n",
    "    - An interesting fact is that the population is split in half between borrowers **owning a house** and not owning a house.\n",
    "    - The borrower's **occupation** is not detailed enough for most of the population, we can therefore avoid analysing this variable in the next stage.  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bivariate Exploration\n",
    "\n",
    "> In this section, we will investigate relationships between pairs of variables. I would like to get some further insights mainly on:\n",
    "- the relationship between credit rating and other variables\n",
    "- the variables that give a higher likelihood of defaulted/charged off loans\n",
    " \n",
    "### 1) Loan Status vs Score\n",
    "> Let's see if there is a relationship between defaulting on a loan or getting the loan charged off and the Prosper Score. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_status = df_loans[df_loans.LoanStatus.notnull()]\n",
    "\n",
    "#store totals per Prosper Score and Loan Status\n",
    "df_score_status = pd.DataFrame(df_status.groupby(['ProsperScore', 'LoanStatus'])['ListingNumber'].count()).reset_index()\n",
    "\n",
    "#store totals per Prosper Score\n",
    "df_score_aggregate = pd.DataFrame(df_score_status.groupby('ProsperScore')['ListingNumber'].sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "#merge\n",
    "df_score_status = df_score_status.merge(df_score_aggregate, on = 'ProsperScore', how = 'left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "#get proportions\n",
    "df_score_status['Proportion'] = df_score_status['ListingNumber_x']/df_score_status['ListingNumber_y']\n",
    "\n",
    "#for visual\n",
    "df_score_status_visual = df_score_status[df_score_status['LoanStatus'].isin(['Chargedoff', 'Defaulted'])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ProsperScore</th>\n",
       "      <th>LoanStatus</th>\n",
       "      <th>ListingNumber_x</th>\n",
       "      <th>ListingNumber_y</th>\n",
       "      <th>Proportion</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>235</td>\n",
       "      <td>992</td>\n",
       "      <td>0.236895</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>37</td>\n",
       "      <td>992</td>\n",
       "      <td>0.037298</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>2.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>377</td>\n",
       "      <td>5766</td>\n",
       "      <td>0.065383</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>2.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>70</td>\n",
       "      <td>5766</td>\n",
       "      <td>0.012140</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>3.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>414</td>\n",
       "      <td>7642</td>\n",
       "      <td>0.054174</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>3.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>75</td>\n",
       "      <td>7642</td>\n",
       "      <td>0.009814</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>4.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>784</td>\n",
       "      <td>12595</td>\n",
       "      <td>0.062247</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>4.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>119</td>\n",
       "      <td>12595</td>\n",
       "      <td>0.009448</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>43</th>\n",
       "      <td>5.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>907</td>\n",
       "      <td>9813</td>\n",
       "      <td>0.092428</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>46</th>\n",
       "      <td>5.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>165</td>\n",
       "      <td>9813</td>\n",
       "      <td>0.016814</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>54</th>\n",
       "      <td>6.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>925</td>\n",
       "      <td>12278</td>\n",
       "      <td>0.075338</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>57</th>\n",
       "      <td>6.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>173</td>\n",
       "      <td>12278</td>\n",
       "      <td>0.014090</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>65</th>\n",
       "      <td>7.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>662</td>\n",
       "      <td>10597</td>\n",
       "      <td>0.062471</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>68</th>\n",
       "      <td>7.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>135</td>\n",
       "      <td>10597</td>\n",
       "      <td>0.012739</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>76</th>\n",
       "      <td>8.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>666</td>\n",
       "      <td>12053</td>\n",
       "      <td>0.055256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>79</th>\n",
       "      <td>8.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>139</td>\n",
       "      <td>12053</td>\n",
       "      <td>0.011532</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>87</th>\n",
       "      <td>9.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>309</td>\n",
       "      <td>6911</td>\n",
       "      <td>0.044711</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>90</th>\n",
       "      <td>9.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>71</td>\n",
       "      <td>6911</td>\n",
       "      <td>0.010273</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>97</th>\n",
       "      <td>10.0</td>\n",
       "      <td>Chargedoff</td>\n",
       "      <td>57</td>\n",
       "      <td>4750</td>\n",
       "      <td>0.012000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100</th>\n",
       "      <td>10.0</td>\n",
       "      <td>Defaulted</td>\n",
       "      <td>20</td>\n",
       "      <td>4750</td>\n",
       "      <td>0.004211</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     ProsperScore  LoanStatus  ListingNumber_x  ListingNumber_y  Proportion\n",
       "0             1.0  Chargedoff              235              992    0.236895\n",
       "3             1.0   Defaulted               37              992    0.037298\n",
       "10            2.0  Chargedoff              377             5766    0.065383\n",
       "13            2.0   Defaulted               70             5766    0.012140\n",
       "21            3.0  Chargedoff              414             7642    0.054174\n",
       "24            3.0   Defaulted               75             7642    0.009814\n",
       "32            4.0  Chargedoff              784            12595    0.062247\n",
       "35            4.0   Defaulted              119            12595    0.009448\n",
       "43            5.0  Chargedoff              907             9813    0.092428\n",
       "46            5.0   Defaulted              165             9813    0.016814\n",
       "54            6.0  Chargedoff              925            12278    0.075338\n",
       "57            6.0   Defaulted              173            12278    0.014090\n",
       "65            7.0  Chargedoff              662            10597    0.062471\n",
       "68            7.0   Defaulted              135            10597    0.012739\n",
       "76            8.0  Chargedoff              666            12053    0.055256\n",
       "79            8.0   Defaulted              139            12053    0.011532\n",
       "87            9.0  Chargedoff              309             6911    0.044711\n",
       "90            9.0   Defaulted               71             6911    0.010273\n",
       "97           10.0  Chargedoff               57             4750    0.012000\n",
       "100          10.0   Defaulted               20             4750    0.004211"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_score_status_visual"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmsAAAFACAYAAADjzzuMAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3XuUVdWZ7/3vIxgxoqCCGRFQMFHbCwgIQqLiBSMaDeqJNmpHse20IWkcehzxPdpnHEXS6dcEPaHNRUXFaGJCDLaKNEkwEaPpqAGEqIg2XohWq0ArXhBRwef8sRfVZVHABmpTaxffzxh71NpzzbXWnLsu/JjrMiMzkSRJUjlt19YNkCRJ0voZ1iRJkkrMsCZJklRihjVJkqQSM6xJkiSVmGFNkiSpxAxrkiRJJWZYkyRJKjHDmiRJUol1bOsGtJZu3bpl796927oZkiRJGzV37tz/yszu1dRtN2Gtd+/ezJkzp62bIUmStFER8Zdq63oaVJIkqcQMa5IkSSVmWJMkSSqxdnPNmiRJqt6HH35IQ0MDq1atauumtGudOnWiZ8+ebL/99pu9D8OaJEnboIaGBnbeeWd69+5NRLR1c9qlzOT111+noaGBPn36bPZ+PA0qSdI2aNWqVey+++4GtRqKCHbfffctHr00rEmStI0yqNVea3zGhjVJkqQSM6xJkqQt0rlz55ru/9vf/jYHHXQQ/fr1o3///jz22GMATJw4kZUrV250+2rrlZVhTZIkldYjjzzC9OnTefzxx3niiSf47W9/S69evQDDmiRJ0mb7y1/+wvDhw+nXrx/Dhw/npZdeAuC+++5jyJAhDBgwgOOOO44lS5YAMG7cOM4//3yOPvpo9tlnH6677joAXn31Vbp168YOO+wAQLdu3dhzzz257rrreOWVVzjmmGM45phjAPj617/OoEGDOOigg7jyyisBWqzXdCRw6tSpnHfeeQD88pe/5OCDD+aQQw5h2LBhtf+QqhSZ2dZtaBWDBg3KluYGPfTS22tyvLkTzq3JfiVJ2hoWLlzIAQcc0Cr76ty5MytWrPhY2Ze+9CVOP/10Ro8ezeTJk5k2bRr33HMPy5cvp2vXrkQEN998MwsXLuTaa69l3LhxzJw5k1mzZvHOO++w//7789prr/H+++9zxBFHsHLlSo477jhGjRrFUUcdBfz3vODdunUD4I033mC33XZjzZo1DB8+nOuuu45+/fqtU69pe6dOncr06dP58Y9/TN++ffn1r39Njx49ePPNN+natWurfD4tfdYRMTczB1WzvSNrkiSp1T3yyCOcffbZAJxzzjn84Q9/ACrPdxsxYgR9+/ZlwoQJLFiwoHGbk046iR122IFu3bqxxx57sGTJEjp37szcuXOZNGkS3bt3Z9SoUfz4xz9u8Zh33nknAwcOZMCAASxYsICnn356k9p8+OGHc95553HTTTexZs2azet4DRjWJElSza19hMWFF17I2LFjefLJJ7nxxhs/9gyytac6ATp06MDq1asbl48++miuuuoqfvCDH3DXXXets/8XX3yRa665ht/97nc88cQTnHTSSet9vlnTx2k0rXPDDTfwT//0T7z88sv079+f119/fcs63UoMa5IkqdV9/vOfZ8qUKQDccccdHHHEEQC89dZb9OjRA4Dbbrtto/t59tlnWbRoUeP7+fPns/feewOw884788477wDw9ttvs9NOO9GlSxeWLFnCr371q8ZtmtYD+NSnPsXChQv56KOPuPvuuxvLn3/+eYYMGcL48ePp1q0bL7/88uZ2v1U53ZQkSdoiK1eupGfPno3vL7nkEq677jrOP/98JkyYQPfu3bn11luByo0EZ5xxBj169GDo0KG8+OKLG9z3ihUruPDCC3nzzTfp2LEjn/3sZ5k0aRIAF1xwASeeeCKf/vSnmTVrFgMGDOCggw5in3324fDDD2/cR/N6V199NSeffDK9evXi4IMPbrx+7dJLL2XRokVkJsOHD+eQQw5p7Y9qs3iDwWbyBgNJUj1rzRsMtGHeYCBJktSOGdYkSZJKzLAmSZJUYoY1SZKkEjOsSZIklZhhTZIkqcR8zpokSWr1R11V84ir1157jYsvvpjZs2ezww470Lt3b0499VSmTZvG9OnTW7U9m2LcuHF07tyZb37zm+ut88wzz3DmmWcSEUydOpV/+7d/4/rrr2fgwIHccccdrdoeR9YkSdJWl5mcdtppHH300Tz//PM8/fTT/PM//zNLlizZov2unaKq1u655x5OOeUU5s2bx2c+8xl+9KMfMWPGjFYPamBYkyRJbWDWrFlsv/32jBkzprGsf//+HHnkkaxYsYLTTz+dv/qrv+Jv/uZvWPsA//HjxzN48GAOPvhgLrjggsbyo48+mn/8x3/kqKOO4l/+5V94/vnnGTp0KIMHD+aKK66gc+fOjceYMGECgwcPpl+/flx55ZWN5d/+9rfZf//9Oe6443j22Wcby+fPn8/QoUPp168fp512GsuXL2fGjBlMnDiRm2++mWOOOYYxY8bwwgsvMHLkSL73ve+1+mflaVBJkrTVPfXUUxx66KEtrps3bx4LFixgzz335PDDD+ff//3fOeKIIxg7dixXXHEFAOeccw7Tp0/nS1/6EgBvvvkmv//97wE4+eSTueiiizjrrLO44YYbGvc7c+ZMFi1axJ/+9Ccyk5EjR/LQQw+x0047MWXKFObNm8fq1asZOHBgY9vOPfdcvv/973PUUUdxxRVXcNVVVzFx4kTGjBnzsVOlv/71r5k1axbdunVr9c/KkTVJklQqhx12GD179mS77bajf//+LF68GKiMxg0ZMoS+ffvywAMPsGDBgsZtRo0a1bj8yCOPcMYZZwBw9tlnN5bPnDmTmTNnMmDAAAYOHMgzzzzDokWLePjhhznttNP45Cc/yS677MLIkSOByqTzb775JkcddRQAo0eP5qGHHqp199fhyJokSdrqDjroIKZOndriuh122KFxuUOHDqxevZpVq1bxjW98gzlz5tCrVy/GjRvHqlWrGuvttNNOGz1mZnL55Zfzta997WPlEydOJCI2sye158iaJEna6o499ljef/99brrppsay2bNnN57KbG5tMOvWrRsrVqxYb9ADGDp0KHfddRcAU6ZMaSwfMWIEkydPZsWKFQD853/+J0uXLmXYsGHcfffdvPfee7zzzjvcd999AHTp0oVdd92Vhx9+GICf/OQnjaNsW5Mja5IkqapHbbSmiODuu+/m4osv5uqrr6ZTp06Nj+5oSdeuXfn7v/97+vbtS+/evRk8ePB69z1x4kS+8pWvcO2113LSSSfRpUsXAI4//ngWLlzI5z73OQA6d+7MT3/6UwYOHMioUaPo378/e++9N0ceeWTjvm677TbGjBnDypUr2Weffbj11ltb8VOoTqy9k6LeDRo0KOfMmbNOeWs/N2atrf1DLUlSa1q4cCEHHHBAWzejJlauXMmOO+5IRDBlyhR+/vOfc++997ZZe1r6rCNibmYOqmZ7R9YkSVK7MnfuXMaOHUtm0rVrVyZPntzWTdoihjVJktSuHHnkkfz5z39u62a0Gm8wkCRJKjHDmiRJUokZ1iRJkkrMsCZJklRi3mAgSZJ4aXzfVt3fXlc8udE6HTp0oG/fvnz44Yd07NiR0aNHc/HFF7PddhseS7r00kuZMWMGX/ziF5kwYcImt61z586sWLGCxYsX88c//vFjU1JV47zzzuPkk0/m9NNP3+Rjbw7DmiRJahM77rgj8+fPB2Dp0qWcffbZvPXWW1x11VUb3O7GG29k2bJlH5uWanMsXryYn/3sZ5sc1rY2T4NKkqQ2t8ceezBp0iR+8IMfkJmsWbOGSy+9lMGDB9OvXz9uvPFGAEaOHMm7777LkCFD+MUvfsF9993HkCFDGDBgAMcddxxLliwBYNy4cVxzzTWN+z/44IMbJ4Rf67LLLuPhhx+mf//+fO9731vvMTOTsWPHcuCBB3LSSSexdOnSrfOhFBxZkyRJpbDPPvvw0UcfsXTpUu699166dOnC7Nmzef/99zn88MM5/vjjmTZtGp07d24ckVu+fDmPPvooEcHNN9/Md7/7Xa699tqqjnf11VdzzTXXMH36dAAmTZrU4jHnzZvHs88+y5NPPsmSJUs48MADOf/882v2OTRnWJMkSaWxdhrMmTNn8sQTTzRO2P7WW2+xaNEi+vTp87H6DQ0NjBo1ildffZUPPvhgnfWbYn3HfOihhzjrrLPo0KEDe+65J8cee+xmH2NzGNYkSVIpvPDCC3To0IE99tiDzOT73/8+I0aM2OA2F154IZdccgkjR47kwQcfZNy4cQB07NiRjz76qLHeqlWrNnr89R1zxowZRMSmd6iVeM2aJElqc8uWLWPMmDGMHTuWiGDEiBFcf/31fPjhhwD8x3/8B+++++4627311lv06NEDgNtuu62xvHfv3jz++OMAPP7447z44ovrbLvzzjvzzjvvNL5f3zGHDRvGlClTWLNmDa+++iqzZs1qvY5XwZE1SZJU1aM2Wtt7771H//79Gx/dcc4553DJJZcA8NWvfpXFixczcOBAMpPu3btzzz33rLOPcePGccYZZ9CjRw+GDh3aGMq+/OUvc/vtt9O/f38GDx7Mfvvtt862/fr1o2PHjhxyyCGcd955XHTRRS0e87TTTuOBBx6gb9++7Lfffhx11FG1/WCaibXnhuvdoEGDcs6cOeuUH3rp7TU53twJ59Zkv5IkbQ0LFy7kgAMOaOtmbBNa+qwjYm5mDqpme0+DSpIklVhNw1pEnBARz0bEcxFxWQvrL4mIpyPiiYj4XUTs3WTd6IhYVLxG17KdkiRJZVWzsBYRHYAfAicCBwJnRcSBzarNAwZlZj9gKvDdYtvdgCuBIcBhwJURsWut2ipJ0raovVwKVWat8RnXcmTtMOC5zHwhMz8ApgCnNK2QmbMyc2Xx9lGgZ7E8Arg/M9/IzOXA/cAJNWyrJEnblE6dOvH6668b2GooM3n99dfp1KnTFu2nlneD9gBebvK+gcpI2fr8HfCrDWzbo/kGEXEBcAHAXnvttSVtlSRpm9KzZ08aGhpYtmxZWzelXevUqRM9e/bceMUNqGVYa+npcS3G94j4CjAIWHsvbFXbZuYkYBJU7gbdvGZKkrTt2X777bfoaf/aemp5GrQB6NXkfU/gleaVIuI44H8DIzPz/U3ZVpIkqb2rZVibDewbEX0i4hPAmcC0phUiYgBwI5Wg1nQK+98Ax0fErsWNBccXZZIkSduUmp0GzczVETGWSsjqAEzOzAURMR6Yk5nTgAlAZ+CXxZxbL2XmyMx8IyK+RSXwAYzPzDdq1VZJkqSyqul0U5k5A5jRrOyKJsvHbWDbycDk2rVOkiSp/JzBQJIkqcQMa5IkSSVmWJMkSSoxw5okSVKJGdYkSZJKzLAmSZJUYoY1SZKkEjOsSZIklZhhTZIkqcQMa5IkSSVmWJMkSSoxw5okSVKJGdYkSZJKzLAmSZJUYoY1SZKkEjOsSZIklZhhTZIkqcQMa5IkSSVmWJMkSSoxw5okSVKJGdYkSZJKzLAmSZJUYoY1SZKkEjOsSZIklZhhTZIkqcQMa5IkSSVmWJMkSSoxw5okSVKJGdYkSZJKzLAmSZJUYoY1SZKkEjOsSZIklZhhTZIkqcQMa5IkSSVmWJMkSSoxw5okSVKJGdYkSZJKzLAmSZJUYoY1SZKkEjOsSZIklZhhTZIkqcQMa5IkSSVmWJMkSSoxw5okSVKJGdYkSZJKzLAmSZJUYjUNaxFxQkQ8GxHPRcRlLawfFhGPR8TqiDi92bo1ETG/eE2rZTslSZLKqmOtdhwRHYAfAl8AGoDZETEtM59uUu0l4Dzgmy3s4r3M7F+r9kmSJNWDmoU14DDgucx8ASAipgCnAI1hLTMXF+s+qmE7JEmS6lYtT4P2AF5u8r6hKKtWp4iYExGPRsSpLVWIiAuKOnOWLVu2JW2VJEkqpVqGtWihLDdh+70ycxBwNjAxIj6zzs4yJ2XmoMwc1L17981tpyRJUmnVMqw1AL2avO8JvFLtxpn5SvH1BeBBYEBrNk6SJKke1DKszQb2jYg+EfEJ4Eygqrs6I2LXiNihWO4GHE6Ta90kSZK2FTULa5m5GhgL/AZYCNyZmQsiYnxEjASIiMER0QCcAdwYEQuKzQ8A5kTEn4FZwNXN7iKVJEnaJlR1N2hE7AdcCuzddJvMPHZD22XmDGBGs7IrmizPpnJ6tPl2fwT6VtM2SZKk9qzaR3f8ErgBuAlYU7vmSJIkqalqw9rqzLy+pi2RJEnSOqq9Zu2+iPhGRHw6InZb+6ppyyRJklT1yNro4uulTcoS2Kd1myNJkqSmqgprmdmn1g2RJEnSuqq9G3R74OvAsKLoQeDGzPywRu2SJEkS1Z8GvR7YHvhR8f6couyrtWiUJEmSKqoNa4Mz85Am7x8oHlgrSZKkGqr2btA1TSdSj4h98HlrkiRJNVftyNqlwKyIeAEIKjMZ/G3NWiVJkiSg+rtBfxcR+wL7Uwlrz2Tm+zVtmSRJkjYc1iLi2Mx8ICL+R7NVn4kIMvNfa9g2SZKkbd7GRtaOAh4AvtTCugQMa5IkSTW0wbCWmVcWi+Mz88Wm6yLCB+VKkiTVWLV3g97VQtnU1myIJEmS1rWxa9b+CjgI6NLsurVdgE61bJgkSZI2fs3a/sDJQFc+ft3aO8Df16pRkiRJqtjYNWv3RsR04H9l5j9vpTZJkiSpsNFr1jJzDfCFrdAWSZIkNVPtDAZ/jIgfAL8A3l1bmJmP16RVkiRJAqoPa58vvo5vUpbAsa3bHEmSJDVV7XRTx9S6IZIkSVpXVc9Zi4guEfF/I2JO8bo2IrrUunGSJEnbumofijuZyuM6/rp4vQ3cWqtGSZIkqaLaa9Y+k5lfbvL+qoiYX4sGSZIk6b9VO7L2XkQcsfZNRBwOvFebJkmSJGmtakfWvg7cVlynFsAbwOiatUqSJElA9XeDzgcOiYhdivdv17RVkiRJAqq/G3T3iLgOeBCYFRH/EhG717RlkiRJqvqatSnAMuDLwOnF8i9q1ShJkiRVVHvN2m6Z+a0m7/8pIk6tRYMkSZL036odWZsVEWdGxHbF66+Bf6tlwyRJklR9WPsa8DPgg+I1BbgkIt6JCG82kCRJqpFq7wbdudYNkSRJ0rqqvWaNiBgJDCvePpiZ02vTJEmSJK1V7aM7rgYuAp4uXhcVZZIkSaqhakfWvgj0z8yPACLiNmAecFmtGiZJkqTqbzAA6NpkuUtrN0SSJEnrqnZk7f8H5kXELCpzgw4DLq9ZqyRJkgRUEdYiIoA/AEOBwVTC2v/KzNdq3DZJkqRt3kbDWmZmRNyTmYcC07ZCmyRJklSo9pq1RyNicE1bIkmSpHVUe83aMcCYiFgMvEvlVGhmZr9aNUySJEnVh7UTa9oKSZIktWiDYS0iOgFjgM8CTwK3ZObqrdEwSZIkbfyatduAQVSC2onAtZuy84g4ISKejYjnImKdB+hGxLCIeDwiVkfE6c3WjY6IRcVr9KYcV5Ikqb3Y2GnQAzOzL0BE3AL8qdodR0QH4IfAF4AGYHZETMvMp5tUewk4D/hms213A66kEhQTmFtsu7za40uSJLUHGxtZ+3Dtwmac/jwMeC4zX8jMD4ApwClNK2Tm4sx8Avio2bYjgPsz840ioN0PnLCJx5ckSap7GxtZOyQi3i6WA9ixeL/2btBdNrBtD+DlJu8bgCFVtqulbXtUua0kSVK7scGwlpkdtmDf0dIuW3PbiLgAuABgr732qr5lkiRJdWJTJnLfVA1ArybvewKvtOa2mTkpMwdl5qDu3btvdkMlSZLKqpZhbTawb0T0iYhPAGdS/XRVvwGOj4hdI2JX4PiiTJIkaZtSs7BW3JAwlkrIWgjcmZkLImJ8RIwEiIjBEdEAnAHcGBELim3fAL5FJfDNBsYXZZIkSduUamcw2CyZOQOY0azsiibLs6mc4mxp28nA5Fq2T5IkqexqeRpUkiRJW8iwJkmSVGKGNUmSpBKr6TVrkrZdh156e832PXfCuTXbtySVjSNrkiRJJWZYkyRJKjHDmiRJUokZ1iRJkkrMsCZJklRihjVJkqQSM6xJkiSVmGFNkiSpxAxrkiRJJWZYkyRJKjHDmiRJUokZ1iRJkkrMsCZJklRihjVJkqQSM6xJkiSVmGFNkiSpxAxrkiRJJdaxrRsgSfXo0Etvr8l+5044tyb7lVS/HFmTJEkqMcOaJElSiRnWJEmSSsywJkmSVGKGNUmSpBIzrEmSJJWYYU2SJKnEDGuSJEklZliTJEkqMcOaJElSiTndVJ1zyhtJkto3R9YkSZJKzLAmSZJUYoY1SZKkEvOaNamN1Op6Q/CaQ205r4eVysORNUmSpBIzrEmSJJWYYU2SJKnEDGuSJEklZliTJEkqMcOaJElSifnoDknSNsdHk6ieOLImSZJUYo6sqdT8368kaVtX05G1iDghIp6NiOci4rIW1u8QEb8o1j8WEb2L8t4R8V5EzC9eN9SynZIkSWVVs5G1iOgA/BD4AtAAzI6IaZn5dJNqfwcsz8zPRsSZwHeAUcW65zOzf63aJ0mSVA9qObJ2GPBcZr6QmR8AU4BTmtU5BbitWJ4KDI+IqGGbJEmS6kotw1oP4OUm7xuKshbrZOZq4C1g92Jdn4iYFxG/j4gjWzpARFwQEXMiYs6yZctat/WSJEklUMuw1tIIWVZZ51Vgr8wcAFwC/CwidlmnYuakzByUmYO6d+++xQ2WJEkqm1qGtQagV5P3PYFX1lcnIjoCXYA3MvP9zHwdIDPnAs8D+9WwrZIkSaVUy7A2G9g3IvpExCeAM4FpzepMA0YXy6cDD2RmRkT34gYFImIfYF/ghRq2VZIkqZRqdjdoZq6OiLHAb4AOwOTMXBAR44E5mTkNuAX4SUQ8B7xBJdABDAPGR8RqYA0wJjPfqFVbJUmSyqqmD8XNzBnAjGZlVzRZXgWc0cJ2dwF31bJtkiRJ9cAZDCRJakdqNfMLOPtLW3FuUEmSpBIzrEmSJJWYYU2SJKnEDGuSJEklZliTJEkqMcOaJElSiRnWJEmSSsznrG2ml8b3rdm+97riyZrtW5Ik1RdH1iRJkkrMsCZJklRihjVJkqQSM6xJkiSVmGFNkiSpxAxrkiRJJWZYkyRJKjHDmiRJUokZ1iRJkkrMsCZJklRihjVJkqQSM6xJkiSVmGFNkiSpxAxrkiRJJWZYkyRJKjHDmiRJUokZ1iRJkkrMsCZJklRihjVJkqQSM6xJkiSVmGFNkiSpxDq2dQMkaVO9NL5vTfa71xVP1mS/krQlHFmTJEkqMcOaJElSiXkaVJJKpFaneMHTvFK9cmRNkiSpxBxZkyRtNd4cIm06w5pa5B9USdp0/u2svUMvvb1m+5474dya7XtLeBpUkiSpxBxZ0zapvf/vt733T1Lb8G9L23BkTZIkqcQMa5IkSSVmWJMkSSoxw5okSVKJGdYkSZJKzLtBJUmSKO/drjUdWYuIEyLi2Yh4LiIua2H9DhHxi2L9YxHRu8m6y4vyZyNiRC3bKUmSVFY1C2sR0QH4IXAicCBwVkQc2Kza3wHLM/OzwPeA7xTbHgicCRwEnAD8qNifJEnSNqWWI2uHAc9l5guZ+QEwBTilWZ1TgNuK5anA8IiIonxKZr6fmS8CzxX7kyRJ2qbUMqz1AF5u8r6hKGuxTmauBt4Cdq9yW0mSpHYvMrM2O444AxiRmV8t3p8DHJaZFzaps6Co01C8f57KCNp44JHM/GlRfgswIzPvanaMC4ALirf7A8/WpDMt6wb811Y83tZm/+qb/atf7blvYP/qnf1rPXtnZvdqKtbybtAGoFeT9z2BV9ZTpyEiOgJdgDeq3JbMnARMasU2Vy0i5mTmoLY49tZg/+qb/atf7blvYP/qnf1rG7U8DTob2Dci+kTEJ6jcMDCtWZ1pwOhi+XTggawM9U0DzizuFu0D7Av8qYZtlSRJKqWajaxl5uqIGAv8BugATM7MBRExHpiTmdOAW4CfRMRzVEbUziy2XRARdwJPA6uBf8jMNbVqqyRJUlnV9KG4mTkDmNGs7Iomy6uAM9az7beBb9eyfVuoTU6/bkX2r77Zv/rVnvsG9q/e2b82ULMbDCRJkrTlnBtUkiSpxAxrkiRJJWZY24iImBwRSyPiqfWsj4i4rpjH9ImIGLi127i5IqJXRMyKiIURsSAiLmqhTj33r1NE/Cki/lz076oW6qx3ftp6EBEdImJeRExvYV29921xRDwZEfMjYk4L6+v2ZxMgIrpGxNSIeKb4Hfxcs/V127+I2L/4vq19vR0RFzerU7f9A4iI/1n8XXkqIn4eEZ2ara/b37+IuKjo14Lm37difd1971r6tzwidouI+yNiUfF11/VsO7qosygiRrdUp+Yy09cGXsAwYCDw1HrWfxH4FRDAUOCxtm7zJvTt08DAYnln4D+AA9tR/wLoXCxvDzwGDG1W5xvADcXymcAv2rrdm9jHS4CfAdNbWFfvfVsMdNvA+rr92Szafxvw1WL5E0DX9tS/Jv3oALxG5QGg7aJ/VGbUeRHYsXh/J3Beszp1+fsHHAw8BXySyk2IvwX2rffvXUv/lgPfBS4rli8DvtPCdrsBLxRfdy2Wd93a7XdkbSMy8yEqjxVZn1OA27PiUaBrRHx667Ruy2Tmq5n5eLH8DrCQdaf1quf+ZWauKN5uX7ya31GzvvlpSy8iegInATevp0rd9q1KdfuzGRG7UPnH4xaAzPwgM99sVq1u+9fMcOD5zPxLs/J6719HYMeoPND9k6z74PZ6/f07AHg0M1dmZRrI3wOnNatTd9+79fxb3vR7dBtwagubjgDuz8w3MnM5cD9wQs0auh6GtS3XLuYxLYboB1AZfWqqrvtXnCacDyyl8gu33v7lx+enrQcTgf8P+Gg96+u5b1AJ1jMjYm5UppZrrp5/NvcBlgG3Fqexb46InZrVqef+NXUm8PMWyuu2f5n5n8A1wEvAq8BbmTmzWbV6/f17ChgWEbtHxCepjKL1alanbr93zXwqM1+FyuAFsEcLdUrRV8Palmvpf0p19TyUiOgM3AVcnJlvN1/dwiavLHX1AAAGGElEQVR107/MXJOZ/alMWXZYRBzcrEpd9i8iTgaWZubcDVVroaz0fWvi8MwcCJwI/ENEDGu2vp7715HKKZnrM3MA8C6V0zBN1XP/AIjK7DUjgV+2tLqFsrroX3Ft0ylAH2BPYKeI+Erzai1sWvr+ZeZC4DtURpB+DfyZysPpm6rLvm2mUvTVsLblqprHtKwiYnsqQe2OzPzXFqrUdf/WKk4xPci6w9eN/YuPz09bdocDIyNiMTAFODYiftqsTr32DYDMfKX4uhS4GzisWZV6/tlsABqajPROpRLemtep1/6tdSLweGYuaWFdPffvOODFzFyWmR8C/wp8vlmduv39y8xbMnNgZg6j0uZFzarU8/euqSVrT98WX5e2UKcUfTWsbblpwLnF3TFDqQyHv9rWjapGcf3ELcDCzPy/66lWz/3rHhFdi+UdqfyBfaZZtfXNT1tqmXl5ZvbMzN5UTjM9kJnN/2dfl30DiIidImLntcvA8VROzzRVtz+bmfka8HJE7F8UDacyvV5Tddu/Js6i5VOgUN/9ewkYGhGfLP6ODqdyzW9T9fz7t0fxdS/gf7Du97Cev3dNNf0ejQbubaHOb4DjI2LXYkT1+KJsq6rpdFPtQUT8HDga6BYRDcCVVC5UJzNvoDKd1heB54CVwN+2TUs3y+HAOcCTxXVdAP8I7AXton+fBm6LiA5U/mNyZ2ZOjyrmp61X7ahvnwLuLq7H7gj8LDN/HRFjoF38bAJcCNxRnCp8Afjb9tS/4nqnLwBfa1LWLvqXmY9FxFTgcSqnCOcBk9rR799dEbE78CGVubmX1/v3bj3/ll8N3BkRf0clgJ9R1B0EjMnMr2bmGxHxLWB2savxmbnVR0idbkqSJKnEPA0qSZJUYoY1SZKkEjOsSZIklZhhTZIkqcQMa5IkSSVmWJNUehGxJiLmR8RTEfHL4rEQbd2m7SLiuqJNT0bE7Ijo09btktT+GNYk1YP3MrN/Zh4MfACMabqyeDjnVvt7VjyRfhSVqYb6ZWZfKpNdN5+MfXP2K0kfY1iTVG8eBj4bEb0jYmFE/IjKw0l7RcRZxSjXUxHxHYCI6BARP24yAvY/i/IHI2JiRPyxWHdYUb5TREwuRsrmRcQpRfl5xajefcBMKg9dfjUzPwLIzIbMXF7UPSEiHo+IP0fE74qy3SLinoh4IiIejYh+Rfm4iJgUETOB24v2TiiO/0REfA1J2zT/FyepbhQjTydSmWAaYH/gbzPzGxGxJ5UJqA8FlgMzI+JU4GWgRzEqx9opyAo7ZebnozJJ/GTgYOB/U5ka6Pyi7p8i4rdF/c9RGUl7IyJ6An+IiCOB3wE/zcx5EdEduAkYlpkvRsRuxbZXAfMy89SIOBa4HehfrDsUOCIz34uIC6hM3zM4InYA/j0iZmbmi632QUqqK46sSaoHOxZTos2hMi3MLUX5XzLz0WJ5MPBgMbn2auAOYBiVqZz2iYjvR8QJwNtN9vtzgMx8CNilCGfHA5cVx3sQ6EQxBRtw/9qpZjKzgUpYvBz4CPhdRAwHhgIPrQ1XTaamOQL4SVH2ALB7RHQp1k3LzPeK5eOpzLs4H3gM2B3YdzM/N0ntgCNrkurBe5nZv2lBMW/ou02LWtqwmNfwEGAE8A/AXwPnr13dvHqxny9n5rPNjjek2fHIzPeBXwG/ioglwKnA/S3sd33tW1uveT8uzMytPlm0pHJyZE1Se/EYcFREdIuIDsBZwO8johuwXWbeBfwfYGCTbUYBRMQRVE49vgX8BrgwijQYEQNaOlhEDCxOvVLc3NAP+AvwSNGOPsW6tadBHwL+pig7GvivzHy7+X6L4389IrYv6u4XETttzgciqX1wZE1Su5CZr0bE5cAsKqNTMzLz3mJU7dYmd4te3mSz5RHxR2AX/nu07VvAROCJIrAtBk5u4ZB7ADcV15UB/An4QWauKq47+9fimEuBLwDjinY8AawERq+nKzcDvYHHi+MvozJiJ2kbFZktjdZLUvsWEQ8C38zMOW3dFknaEE+DSpIklZgja5IkSSXmyJokSVKJGdYkSZJKzLAmSZJUYoY1SZKkEjOsSZIkldj/A17RA6RsiXFkAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = df_score_status_visual, x = 'ProsperScore', y = 'Proportion', hue = 'LoanStatus');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> This is telling us that more than 20% of the borrowers with a Prosper score of 1 have their loan charged off and almost 5% defaulted. These percentages go decreasing and are the lowest for borrowers with credit score equal to 10, which shows that the two features are correlated and people with a lower Prosper score tend to have a higher chance to have their loans charge off or default.  \n",
    "\n",
    "### 2) Loan Status vs Credit Grade/Prosper Rating\n",
    "> We will now perform a similar analysis, looking at the relationship between the amount of charged off/defaulted loans and credit rating."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_creditGrade = df_loans[df_loans.CreditGrade.notnull()]\n",
    "df_creditGrade = df_creditGrade[df_creditGrade.LoanStatus.notnull()]\n",
    "\n",
    "#store totals per Credit Grade and Loan Status\n",
    "df_grade_status = pd.DataFrame(df_creditGrade.groupby(['CreditGrade', 'LoanStatus'])['ListingNumber'].count()).reset_index()\n",
    "\n",
    "#store totals per Credit Grade\n",
    "df_grade_aggregate = pd.DataFrame(df_grade_status.groupby('CreditGrade')['ListingNumber'].sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "#merge\n",
    "df_grade_status = df_grade_status.merge(df_grade_aggregate, on = 'CreditGrade', how = 'left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "#get proportions\n",
    "df_grade_status['Proportion'] = df_grade_status['ListingNumber_x']/df_grade_status['ListingNumber_y']\n",
    "\n",
    "#for visual\n",
    "df_grade_status_visual = df_grade_status[df_grade_status['LoanStatus'].isin(['Chargedoff', 'Defaulted'])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmQAAAFACAYAAAASxGABAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3X2clVW99/HPz8HAI4gpWMmDiKnJkwOCUKiokA+pGIUHsVSOp4i68ejxlvtYnRciPXmLHsk0FQ3FnrDwaEhUnBLTygoQkkY0fHbSkNsUQUQF1/3H7Jm2w8DMwOxZG+bzfr3mdfZ17Wtf6zfrCHxb19prRUoJSZIk5bNH7gIkSZLaOgOZJElSZgYySZKkzAxkkiRJmRnIJEmSMjOQSZIkZWYgkyRJysxAJkmSlJmBTJIkKbN2uQtori5duqRevXrlLkOSJKlRy5Yt+38ppa6NXbfLBbJevXqxdOnS3GVIkiQ1KiKebcp1PrKUJEnKzEAmSZKUmYFMkiQps11uDpkkSWqat99+m+rqajZt2pS7lN1ehw4d6N69O3vuuecOfd5AJknSbqq6uppOnTrRq1cvIiJ3ObutlBIvv/wy1dXVHHzwwTt0Dx9ZSpK0m9q0aRP777+/YazEIoL9999/p0YiDWSSJO3GDGOtY2f72UAmSZKUmYFMkiQ1qmPHjiW9/9e+9jX69u3LgAEDqKys5A9/+AMAM2fOZOPGjY1+vqnXlSsDmSRJyuqhhx5iwYIFPPzwwzzyyCP88pe/pEePHoCBTJIkabueffZZRo4cyYABAxg5ciTPPfccAPfeey9Dhw5l4MCBjBo1ijVr1gAwbdo0LrjgAo4//nh69+7NddddB8CLL75Ily5daN++PQBdunThwAMP5LrrruOFF17ghBNO4IQTTgDg85//PIMHD6Zv375cfvnlAA1eVzyiN2/ePCZMmADAj3/8Y/r168eRRx7JcccdV/pOaqJIKeWuoVkGDx6cymUvy+em92/V9npOXdmq7UmSdm2rVq3iiCOOaJF7dezYkQ0bNrzr3BlnnMHYsWM5//zzmT17NvPnz+eee+7hlVdeYd999yUiuPXWW1m1ahXXXHMN06ZNY9GiRSxevJj169dz+OGH87e//Y0333yTY445ho0bNzJq1CjGjRvHiBEjgH/sYd2lSxcA/v73v7PffvuxZcsWRo4cyXXXXceAAQO2uq643nnz5rFgwQJuv/12+vfvz89//nO6devGq6++yr777tsi/QMN93dELEspDW7ss46QSZKkHfLQQw9xzjnnAHDuuefym9/8BqhZ/+zkk0+mf//+zJgxg6qqqrrPnHbaabRv354uXbpwwAEHsGbNGjp27MiyZcuYNWsWXbt2Zdy4cdx+++0NtvmjH/2IQYMGMXDgQKqqqnj00UebVfPw4cOZMGECt9xyC1u2bNmxX7wEDGSSJKlF1C79cOGFFzJ58mRWrlzJzTff/K71uWofSwJUVFSwefPmutfHH388V1xxBddffz133XXXVvd/+umnufrqq/nVr37FI488wmmnnbbNtb+Kl6Eovuamm27iq1/9Ks8//zyVlZW8/PLLO/dLtxADmSRJ2iEf+chHmDt3LgDf//73OeaYYwBYt24d3bp1A2DOnDmN3ufxxx9n9erVdccrVqzgoIMOAqBTp06sX78egNdee429996bzp07s2bNGn72s5/Vfab4OoD3ve99rFq1infeeYe777677vyTTz7J0KFDmT59Ol26dOH555/f0V+/Rbl1kiRJatTGjRvp3r173fEll1zCddddxwUXXMCMGTPo2rUrt912G1Azef+ss86iW7duDBs2jKeffnq7996wYQMXXnghr776Ku3ateODH/wgs2bNAmDixImceuqpfOADH2Dx4sUMHDiQvn370rt3b4YPH153j/rXXXnllZx++un06NGDfv361c0nmzJlCqtXryalxMiRIznyyCNbuqt2iJP6d4KT+iVJ5awlJ/WrcU7qlyRJ2oUZyCRJkjIzkEmSJGVmIJMkScrMQCZJkpSZgUySJCkz1yGTJKmNOGrKHS16v2UzzmvSdX/729+4+OKLWbJkCe3bt6dXr158/OMfZ/78+SxYsKBFa2qOadOm0bFjRy699NJtXvPYY49x9tlnExHMmzePn/70p9x4440MGjSI73//+y1WiyNkkiSpZFJKjBkzhuOPP54nn3ySRx99lK9//eusWbNmp+5bu+VSqd1zzz2ceeaZLF++nEMOOYRvf/vbLFy4sEXDGBjIJElSCS1evJg999yTSZMm1Z2rrKzk2GOPZcOGDYwdO5YPfehDfOpTn6J2sfrp06czZMgQ+vXrx8SJE+vOH3/88XzpS19ixIgRfPOb3+TJJ59k2LBhDBkyhKlTp9KxY8e6NmbMmMGQIUMYMGAAl19+ed35r33taxx++OGMGjWKxx9/vO78ihUrGDZsGAMGDGDMmDG88sorLFy4kJkzZ3LrrbdywgknMGnSJJ566ilGjx7Ntdde26L95CNLSZJUMn/+85856qijGnxv+fLlVFVVceCBBzJ8+HB++9vfcswxxzB58mSmTp0KwLnnnsuCBQs444wzAHj11Vf59a9/DcDpp5/ORRddxPjx47npppvq7rto0SJWr17NH//4R1JKjB49mgceeIC9996buXPnsnz5cjZv3sygQYPqajvvvPP41re+xYgRI5g6dSpXXHEFM2fOZNKkSe96rPnzn/+cxYsX06VLlxbtJ0fIJElSFkcffTTdu3dnjz32oLKykmeeeQaoGVUbOnQo/fv357777qOqqqruM+PGjat7/dBDD3HWWWcBcM4559SdX7RoEYsWLWLgwIEMGjSIxx57jNWrV/Pggw8yZswY/umf/ol99tmH0aNHAzWbob/66quMGDECgPPPP58HHnig1L/+uzhCJkmSSqZv377Mmzevwffat29f97qiooLNmzezadMmvvCFL7B06VJ69OjBtGnT2LRpU911e++9d6NtppT44he/yOc+97l3nZ85cyYRsYO/SWk5QiZJkkrmxBNP5M033+SWW26pO7dkyZK6x4711YavLl26sGHDhm2GOYBhw4Zx1113ATB37ty68yeffDKzZ89mw4YNAPz1r3/lpZde4rjjjuPuu+/mjTfeYP369dx7770AdO7cmfe+9708+OCDAHz3u9+tGy1rLY6QSZLURjR1mYqWFBHcfffdXHzxxVx55ZV06NChbtmLhuy777589rOfpX///vTq1YshQ4Zs894zZ87k05/+NNdccw2nnXYanTt3BuCkk05i1apVfPjDHwagY8eOfO9732PQoEGMGzeOyspKDjroII499ti6e82ZM4dJkyaxceNGevfuzW233daCvdC4qP3mwq5i8ODBaenSpbnLAOC56f1btb2eU1e2anuSpF3bqlWrOOKII3KXUTIbN25kr732IiKYO3cuP/zhD/nJT36SrZ6G+jsilqWUBjf2WUfIJEnSLmnZsmVMnjyZlBL77rsvs2fPzl3SDjOQSZKkXdKxxx7Ln/70p9xltAgn9UuSJGVmIJMkScrMQCZJkpSZgUySJCmzkk7qj4hTgG8CFcCtKaUrt3HdWODHwJCUUnmsaSFJ0m6mpZdraspyTBUVFfTv35+3336bdu3acf7553PxxRezxx7bHxOaMmUKCxcu5GMf+xgzZsxodm0dO3Zkw4YNPPPMM/zud79719ZKTTFhwgROP/10xo4d2+y2d0TJAllEVAA3AB8FqoElETE/pfRoves6Af8G/KFUtUiSpDz22msvVqxYAcBLL73EOeecw7p167jiiiu2+7mbb76ZtWvXvmt7pR3xzDPP8IMf/KDZgay1lfKR5dHAEymlp1JKbwFzgTMbuO4rwFXApgbekyRJu4kDDjiAWbNmcf3115NSYsuWLUyZMoUhQ4YwYMAAbr75ZgBGjx7N66+/ztChQ7nzzju59957GTp0KAMHDmTUqFGsWbMGgGnTpnH11VfX3b9fv351G5TXuuyyy3jwwQeprKzk2muv3WabKSUmT55Mnz59OO2003jppZdap1MKSvnIshvwfNFxNTC0+IKIGAj0SCktiIhLS1iLJEkqA7179+add97hpZde4ic/+QmdO3dmyZIlvPnmmwwfPpyTTjqJ+fPn07Fjx7qRtVdeeYXf//73RAS33norV111Fddcc02T2rvyyiu5+uqrWbBgAQCzZs1qsM3ly5fz+OOPs3LlStasWUOfPn244IILStYP9ZUykDW0nXrdPk0RsQdwLTCh0RtFTAQmAvTs2bOFypMkSTnUbtu4aNEiHnnkkboNxNetW8fq1as5+OCD33V9dXU148aN48UXX+Stt97a6v3m2FabDzzwAOPHj6eiooIDDzyQE088cYfb2BGlDGTVQI+i4+7AC0XHnYB+wP0RAfB+YH5EjK4/sT+lNAuYBTV7WZawZkmSVEJPPfUUFRUVHHDAAaSU+Na3vsXJJ5+83c9ceOGFXHLJJYwePZr777+fadOmAdCuXTveeeeduus2bWp89tO22ly4cCGFPJJFKeeQLQEOjYiDI+I9wNnA/No3U0rrUkpdUkq9Ukq9gN8DW4UxSZK0e1i7di2TJk1i8uTJRAQnn3wyN954I2+//TYAf/nLX3j99de3+ty6devo1q0bAHPmzKk736tXLx5++GEAHn74YZ5++umtPtupUyfWr19fd7ytNo877jjmzp3Lli1bePHFF1m8eHHL/eJNULIRspTS5oiYDPyCmmUvZqeUqiJiOrA0pTR/+3eQJEktqSnLVLS0N954g8rKyrplL84991wuueQSAD7zmc/wzDPPMGjQIFJKdO3alXvuuWere0ybNo2zzjqLbt26MWzYsLrg9clPfpI77riDyspKhgwZwmGHHbbVZwcMGEC7du048sgjmTBhAhdddFGDbY4ZM4b77ruP/v37c9hhhzFixIjSdkw9Ufscd1cxePDgtHRpeQyitfR6Lo3J8QdJkrTrWrVqFUcccUTuMtqMhvo7IpallAY39llX6pckScrMQCZJkpSZgUySpN3YrjY1aVe1s/1sIJMkaTfVoUMHXn75ZUNZiaWUePnll+nQocMO36Okm4tLkqR8unfvTnV1NWvXrs1dym6vQ4cOdO/efYc/byCTJGk3teeee+7UqvZqPT6ylCRJysxAJkmSlJmBTJIkKTMDmSRJUmYGMkmSpMwMZJIkSZkZyCRJkjIzkEmSJGVmIJMkScrMQCZJkpSZgUySJCkzA5kkSVJmBjJJkqTMDGSSJEmZGcgkSZIya5e7AJXeUVPuaNX2ls04r1XbkyRpV+cImSRJUmYGMkmSpMwMZJIkSZkZyCRJkjIzkEmSJGVmIJMkScrMQCZJkpSZgUySJCkzA5kkSVJmBjJJkqTMDGSSJEmZGcgkSZIyM5BJkiRlZiCTJEnKzEAmSZKUmYFMkiQpMwOZJElSZgYySZKkzAxkkiRJmRnIJEmSMjOQSZIkZWYgkyRJysxAJkmSlJmBTJIkKTMDmSRJUmYGMkmSpMxKGsgi4pSIeDwinoiIyxp4f1JErIyIFRHxm4joU8p6JEmSylHJAllEVAA3AKcCfYDxDQSuH6SU+qeUKoGrgP8qVT2SJEnlqpQjZEcDT6SUnkopvQXMBc4sviCl9FrR4d5AKmE9kiRJZaldCe/dDXi+6LgaGFr/ooj4X8AlwHuAE0tYjyRJUlkq5QhZNHBuqxGwlNINKaVDgP8A/rPBG0VMjIilEbF07dq1LVymJElSXqUMZNVAj6Lj7sAL27l+LvDxht5IKc1KKQ1OKQ3u2rVrC5YoSZKUXykD2RLg0Ig4OCLeA5wNzC++ICIOLTo8DVhdwnokSZLKUsnmkKWUNkfEZOAXQAUwO6VUFRHTgaUppfnA5IgYBbwNvAKcX6p6JEmSylUpJ/WTUloILKx3bmrR64tK2b4kSdKuwJX6JUmSMjOQSZIkZVbSR5ZSOTtqyh2t1tayGee1WluSpF2PI2SSJEmZGcgkSZIyM5BJkiRlZiCTJEnKrEmT+iPiMGAKcFDxZ1JKbgYuSZK0k5r6LcsfAzcBtwBbSleOJElS29PUQLY5pXRjSSuRJElqo5o6h+zeiPhCRHwgIvar/SlpZZIkSW1EU0fIajf9nlJ0LgG9W7YcSZKktqdJgSyldHCpC5EkSWqrmvotyz2BzwPHFU7dD9ycUnq7RHVJkiS1GU19ZHkjsCfw7cLxuYVznylFUZIkSW1JUwPZkJTSkUXH90XEn0pRkCRJUlvT1G9ZbomIQ2oPIqI3rkcmSZLUIpo6QjYFWBwRTwFBzYr9/1KyqiRlcdSUO1q1vWUzzmvV9iSpXDX1W5a/iohDgcOpCWSPpZTeLGllkiRJbcR2A1lEnJhSui8iPlHvrUMigpTSf5ewNkmSpDahsRGyEcB9wBkNvJcAA5kkSdJO2m4gSyldXng5PaX0dPF7EeFisWrQc9P7t2p7PaeubNX2JElqaU39luVdDZyb15KFSJIktVWNzSH7ENAX6FxvHtk+QIdSFiZJktRWNDaH7HDgdGBf3j2PbD3w2VIVtaNa+yv7d3dq1eYkSdJuqrE5ZD+JiAXAf6SUvt5KNUmSJLUpjc4hSyltAT7aCrVIkiS1SU1dqf93EXE9cCfweu3JlNLDJalKkiSpDWlqIPtI4f9OLzqXgBNbthxJkqS2p6lbJ51Q6kIkSZLaqiatQxYRnSPivyJiaeHnmojoXOriJEmS2oKmLgw7m5qlLv658PMacFupipIkSWpLmjqH7JCU0ieLjq+IiBWlKEiSJKmtaeoI2RsRcUztQUQMB94oTUmSJEltS1NHyD4PzCnMGwvg78D5JatKkspIa+8CsmzGea3anqT8mvotyxXAkRGxT+H4tZJWJUmS1IY09VuW+0fEdcD9wOKI+GZE7F/SyiRJktqIps4hmwusBT4JjC28vrNURUmSJLUlTZ1Dtl9K6StFx1+NiI+XoiBJkqS2pqkjZIsj4uyI2KPw88/AT0tZmCRJUlvR1ED2OeAHwFuFn7nAJRGxPiKc4C9JkrQTmvoty06lLkSSJKmtauocMiJiNHBc4fD+lNKC0pQkSZLUtjR12YsrgYuARws/FxXOSZIkaSc1dYTsY0BlSukdgIiYAywHLitVYZIkSW1FUyf1A+xb9LpzSxciSZLUVjV1hOwbwPKIWEzNXpbHAV8sWVWSJEltSKOBLCIC+A0wDBhCTSD7j5TS30pcmyRJUpvQaCBLKaWIuCeldBQwvzk3j4hTgG8CFcCtKaUr671/CfAZYDM12zFdkFJ6tjltSJLyOGrKHa3a3rIZ57Vqe1Jrauocst9HxJDm3DgiKoAbgFOBPsD4iOhT77LlwOCU0gBgHnBVc9qQJEnaHTR1DtkJwKSIeAZ4nZrHlqkQpLblaOCJlNJTABExFziTmmUzoOYGi4uu/z3w6aaXLu06npvev1Xb6zl1Zau2J0naOU0NZKfuwL27Ac8XHVcDQ7dz/b8CP9uBdiRJknZp2w1kEdEBmAR8EFgJfCeltLmJ944GzqVttPNpYDAwYhvvTwQmAvTs2bOJzUuSJO0aGptDNoeaoLSSmlGya5px72qgR9Fxd+CF+hdFxCjgy8DolNKbDd0opTQrpTQ4pTS4a9euzShBkiSp/DX2yLJPSqk/QER8B/hjM+69BDg0Ig4G/gqcDZxTfEFEDARuBk5JKb3UjHtLkiTtNhobIXu79kUzHlUWXz8Z+AWwCvhRSqkqIqYXNioHmAF0BH4cESsiolnLakiSJO0OGhshOzIiXiu8DmCvwnHttyz32d6HU0oLgYX1zk0tej2q+SVLkiTtXrYbyFJKFa1ViCRJUlvVnM3FJUmSVAIGMkmSpMwMZJIkSZkZyCRJkjIzkEmSJGVmIJMkScrMQCZJkpSZgUySJCkzA5kkSVJmBjJJkqTMDGSSJEmZGcgkSZIyM5BJkiRlZiCTJEnKzEAmSZKUmYFMkiQpMwOZJElSZu1yFyBJ0u7kqCl3tFpby2ac12ptqbQcIZMkScrMQCZJkpSZgUySJCkzA5kkSVJmBjJJkqTMDGSSJEmZGcgkSZIyM5BJkiRlZiCTJEnKzEAmSZKUmYFMkiQpMwOZJElSZgYySZKkzAxkkiRJmbXLXYCktuu56f1btb2eU1e2anuS1FSOkEmSJGXmCJkklRlHDqW2xxEySZKkzAxkkiRJmRnIJEmSMjOQSZIkZWYgkyRJysxAJkmSlJmBTJIkKTMDmSRJUmYGMkmSpMxcqV+StEtwBwPtzhwhkyRJysxAJkmSlFlJA1lEnBIRj0fEExFxWQPvHxcRD0fE5ogYW8paJEmSylXJAllEVAA3AKcCfYDxEdGn3mXPAROAH5SqDkmSpHJXykn9RwNPpJSeAoiIucCZwKO1F6SUnim8904J65AkSSprpXxk2Q14vui4unBOkiRJRUoZyKKBc2mHbhQxMSKWRsTStWvX7mRZkiRJ5aWUgawa6FF03B14YUdulFKalVIanFIa3LVr1xYpTpIkqVyUcg7ZEuDQiDgY+CtwNnBOCduTJEll6Kgpd7Rqe8tmnNeq7bWEko2QpZQ2A5OBXwCrgB+llKoiYnpEjAaIiCERUQ2cBdwcEVWlqkeSJKlclXTrpJTSQmBhvXNTi14voeZRpiRJUpvlSv2SJEmZGcgkSZIyM5BJkiRlVtI5ZJIkqXSem96/VdvrOXVlq7bXljhCJkmSlJmBTJIkKTMDmSRJUmYGMkmSpMwMZJIkSZkZyCRJkjIzkEmSJGVmIJMkScrMQCZJkpSZgUySJCkzA5kkSVJmBjJJkqTMDGSSJEmZGcgkSZIyM5BJkiRl1i53AZIkSS3puen9W7W9nlNX7vQ9HCGTJEnKzEAmSZKUmYFMkiQpMwOZJElSZgYySZKkzAxkkiRJmRnIJEmSMjOQSZIkZWYgkyRJysxAJkmSlJmBTJIkKTMDmSRJUmYGMkmSpMwMZJIkSZkZyCRJkjIzkEmSJGVmIJMkScrMQCZJkpSZgUySJCkzA5kkSVJmBjJJkqTMDGSSJEmZGcgkSZIyM5BJkiRlZiCTJEnKzEAmSZKUmYFMkiQpMwOZJElSZiUNZBFxSkQ8HhFPRMRlDbzfPiLuLLz/h4joVcp6JEmSylHJAllEVAA3AKcCfYDxEdGn3mX/CrySUvogcC3wf0tVjyRJUrkq5QjZ0cATKaWnUkpvAXOBM+tdcyYwp/B6HjAyIqKENUmSJJWdUgaybsDzRcfVhXMNXpNS2gysA/YvYU2SJEllJ1JKpblxxFnAySmlzxSOzwWOTildWHRNVeGa6sLxk4VrXq53r4nAxMLh4cDjJSm6+boA/y93EWXIftmafdIw+6Vh9kvD7Jet2ScNK6d+OSil1LWxi9qVsIBqoEfRcXfghW1cUx0R7YDOwN/r3yilNAuYVaI6d1hELE0pDc5dR7mxX7ZmnzTMfmmY/dIw+2Vr9knDdsV+KeUjyyXAoRFxcES8BzgbmF/vmvnA+YXXY4H7UqmG7CRJkspUyUbIUkqbI2Iy8AugApidUqqKiOnA0pTSfOA7wHcj4glqRsbOLlU9kiRJ5aqUjyxJKS0EFtY7N7Xo9SbgrFLWUGJl9xi1TNgvW7NPGma/NMx+aZj9sjX7pGG7XL+UbFK/JEmSmsatkyRJkjIzkEmSJGVmIGuCiEgRcU3R8aURMa3o+LyI+HNEVEXEoxFxaZZCW1lEbKh3PCEiri+8nhYRf42IFYU+GZ+nyrwiYkuhD2p/ttrTta0p6pOqiPhTRFwSEf5dBETE+yNibkQ8WfhzszAiDstdV05F/738KSIejoiP5K6pXETEmMK/Tx/KXUtO2+qHiPj3iNgUEZ1z1dYc/iXYNG8Cn4iILvXfiIhTgYuBk1JKfYFB1Ow4ILg2pVRJzRZZN0fEnrkLyuCNlFJl0c+VuQsqA7V90hf4KPAx4PLMNWVX2DbubuD+lNIhKaU+wJeA9+WtLLva/16OBL4IfCN3QWVkPPAbXKFgW/0wnpoluMa0ekU7wEDWNJup+cbGvzfw3heBS1NKL0DNN0dTSre0ZnHlLqW0GtgIvDd3LSovKaWXqNmFY7L72HIC8HZK6abaEymlFSmlBzPWVG72AV7JXUQ5iIiOwHDgX2nDgWxb/RARhwAdgf+kJpiVvZIue7GbuQF4JCKuqne+H7AsQz3lYK+IWFF0vB9bL/5LRAwCVhf+8W1r6vfRN1JKd2arpgyllJ4qPLI8AFiTu56M2vLfJdtT+2eoA/AB4MTM9ZSLjwM/Tyn9JSL+HhGDUkoP5y4qg231w3jgh8CDwOERcUC5/xtkIGuilNJrEXEH8G/AG7nrKRNvFB5JAjVzyIDirSr+PSI+C/QGTmnl2srFu/pI29TWR8e0bXV/hiLiw8AdEdHPXV0YD8wsvJ5bOG6LgWxb/XA2MCal9E5E/Dc1a57ekKfEpjGQNc9Mav4ffVvRuSrgKOC+LBWVt2tTSldHxCeo+Uv0kMJiwFKdiOgNbAHK+n+9toIqaraQ0zaklB4qzOXtShv+7yUi9qdmpLBfRCRqdsNJEfF/2lJQ3U4/fA84FPifwkyI9wBPUeaBzDlkzZBS+jvwI2qeVdf6BnBVRLwfICLaR8S/5aivXKWU/htYyj/2LZUAiIiuwE3A9W3pH5JtuA9oXxhVBiAihkTEiIw1lZXCt+gqgJdz15LZWOCOlNJBKaVeKaUewNPAMZnram3b6oeZwLTCuV4ppQOBbhFxUNZqG2Ega75rgLpvWxa2h7oB+GVEVFEzB8SRx61NB9ri8gZ71Vv2wm9Z/qNPqoBfAouAKzLXlF0hkI4BPlpY9qIKmAa8kLWw/Or+DAF3AuenlLbkLiqz8dR8I7fYXcA5GWrJaVv90KuB83dT5l9+cOskSZKkzNraaIUkSVLZMZBJkiRlZiCTJEnKzEAmSZKUmYFMkiQpMwOZpLITEe+PiLmF5R8ejYiFEXHYDt5rQkRcX3g9KSLOKzp/YNF17SLi6xGxumiZki/v5O9xfEQs2Jl7SGobXC9LUlkpbDJ+NzAnpXR24Vwl8D7gL4Xjih1Zi6p4425gAvBn/rHO11eB9wP9U0qbIqIAbKsfAAACyElEQVQT8L+3UV+klN5pbvuStC2OkEkqNycAbxeHp5TSCqAiIhZHxA+AlQAR8emI+GNhNOvmiKgonP+XiPhLRPwaGF57n4iYFhGXRsRYavZd/X7hs3sDnwUurN3eK6W0PqU0rfC5XhGxKiK+Tc32aT0i4saIWBoRVRFxRVEbp0TEYxHxG+ATRef3jojZEbEkIpZHxJml6T5JuyIDmaRy04+aHS8acjTw5ZRSn4g4AhgHDC9sPr0F+FREfICalf+HAx8F+tS/SUppHjXbeX2q8NlDgOdSSuu3U9fh1GzTMjCl9GyhjsHAAGBERAyIiA7ALcAZwLHUjLjV+jJwX0ppCDWhc0YhCEqSgUzSLuWPKaWnC69HAkcBSwrb6owEegNDgftTSmtTSm9Rs91OsxRG2FZExPMR0aNw+tmU0u+LLvvniHgYWA70pSb4fQh4OqW0urAV0veKrj8JuKxQ6/1AB6Bnc2uTtHtyDpmkclNFzabBDXm96HVQM8/si8UXRMTHgebuCfcE0DMiOhUeVd4G3BYRf6ZmM+t3tR0RBwOXAkNSSq9ExO3UBCy203YAn0wpPd7M2iS1AY6QSSo39wHtI+KztSciYggwot51vwLGRsQBhWv2i4iDgD8Ax0fE/hGxJ3DWNtpZD3QCSCltBL4DXF947EhhPtp7tvHZfagJaOsi4n3AqYXzjwEHR8QhhePxRZ/5BXBh4UsBRMTA7fSBpDbGQCaprBQe9Y0BPlpY9qIKmMY/vg1Ze92jwH8CiyLiEeB/gA+klF4sXP8Q8EtqJuE35HbgpsKjyb2omeP1IvDniFgOPAjMqd9uoe0/UfOosgqYDfy2cH4TMBH4aWFS/7NFH/sKsCfwSGHk7StN7xVJu7uo+btPkiRJuThCJkmSlJmBTJIkKTMDmSRJUmYGMkmSpMwMZJIkSZkZyCRJkjIzkEmSJGX2/wET4cT58giR0gAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "order_creditGrade = ['NC', 'HR', 'E', 'D', 'C', 'B', 'A', 'AA']\n",
    "\n",
    "sb.barplot(data = df_grade_status_visual, x = 'CreditGrade', y = 'Proportion', hue = 'LoanStatus', order = order_creditGrade);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_prosperRating = df_loans[df_loans['ProsperRating (numeric)'].notnull()]\n",
    "df_prosperRating = df_prosperRating[df_prosperRating.LoanStatus.notnull()]\n",
    "\n",
    "#store totals per Prosper Rating and Loan Status\n",
    "df_rating_status = pd.DataFrame(df_prosperRating.groupby(['ProsperRating (numeric)', 'LoanStatus'])['ListingNumber'].count()).reset_index()\n",
    "\n",
    "#store totals per Prosper Rating\n",
    "df_rating_aggregate = pd.DataFrame(df_rating_status.groupby('ProsperRating (numeric)')['ListingNumber'].sum())\n",
    "\n",
    "#merge\n",
    "df_rating_status = df_rating_status.merge(df_rating_aggregate, on = 'ProsperRating (numeric)', how = 'left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "#get proportions\n",
    "df_rating_status['Proportion'] = df_rating_status['ListingNumber_x']/df_rating_status['ListingNumber_y']\n",
    "\n",
    "#for visual\n",
    "df_rating_status_visual = df_rating_status[df_rating_status['LoanStatus'].isin(['Chargedoff', 'Defaulted'])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAnEAAAFACAYAAADTQyqtAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3Xu8V2WdwPvPVzBoRDEF5yhoaF7yAm4QhFLxmpdU1JPmpfGSNWYNHh1PTNo5KTLVsVFHM0sl81YaFY6KRGolppUZF0lEIlFRd5owXghEUPB7/vitvfu53Zv9A/aPvZd83q/X78Vaz20960mY7zzPWuuJzESSJEnlslFnd0CSJElrziBOkiSphAziJEmSSsggTpIkqYQM4iRJkkrIIE6SJKmEDOIkSZJKyCBOkiSphAziJEmSSqh7Z3dgfejTp08OGDCgs7shSZLUrhkzZvxvZvZtr9wGEcQNGDCA6dOnd3Y3JEmS2hURz9VSzuVUSZKkEjKIkyRJKiGDOEmSpBLaIJ6JkyRJtXn77bdpbGxk+fLlnd2V972ePXvSv39/Nt5447WqbxAnSZKaNTY2summmzJgwAAiorO7876Vmbzyyis0Njay/fbbr1UbLqdKkqRmy5cvZ8sttzSAq7OIYMstt1ynGU+DOEmS9C4GcOvHuo6zQZwkSVIJ1TWIi4jDI2JeRMyPiAtayR8ZETMjYmVEHF+VfmBEzKr6LY+IY4u8myPi2aq8hnregyRJWju9evWqa/vf+MY32H333Rk0aBANDQ08+uijAFx11VUsW7as3fq1luuq6hbERUQ34LvAEcBuwMkRsVuLYs8DZwC3Vydm5tTMbMjMBuAgYBlwf1WRMU35mTmrXvcgSZK6pkceeYTJkyczc+ZMHn/8cX71q1+x7bbbAgZxHWFvYH5mPpOZbwETgGOqC2Tmgsx8HHhnNe0cD/wiM8s7ypIkCYDnnnuOgw8+mEGDBnHwwQfz/PPPA3DPPfcwfPhwBg8ezCGHHMLLL78MwNixYznzzDM54IAD2GGHHbj66qsBeOmll+jTpw89evQAoE+fPmyzzTZcffXVvPjiixx44IEceOCBAHzxi19k6NCh7L777lx88cUArZarnjmcOHEiZ5xxBgA/+9nP2GOPPdhzzz0ZOXJk/QepRpGZ9Wm4sjx6eGZ+vjg/FRiemaNbKXszMDkzJ7aS9wDw35k5uarsx4AVwK+BCzJzRSv1zgLOAthuu+32eu65trch22vMrWt6ex1qxmWnder1JUlqMnfuXHbdddcOaatXr14sXbr0XWlHH300xx9/PKeffjo33ngjkyZN4q677uK1115j8803JyK44YYbmDt3LldccQVjx47l/vvvZ+rUqSxZsoRddtmFv/3tb6xYsYJ9992XZcuWccghh3DiiSey//77A//YM71Pnz4AvPrqq2yxxRasWrWKgw8+mKuvvppBgwa9p1x1fydOnMjkyZO5+eabGThwIPfeey/9+vXj9ddfZ/PNN++Q8YHWxzsiZmTm0Pbq1nMmrrVXLtYoYoyIrYGBwH1VyRcCHwWGAVsAX2mtbmaOz8yhmTm0b9++a3JZSZJUJ4888ginnHIKAKeeeiq//e1vgcr36Q477DAGDhzIZZddxpw5c5rrHHnkkfTo0YM+ffqw1VZb8fLLL9OrVy9mzJjB+PHj6du3LyeeeCI333xzq9f86U9/ypAhQxg8eDBz5szhySefXKM+77PPPpxxxhl8//vfZ9WqVWt343VQzyCuEdi26rw/8OIatvFp4M7MfLspITNfyooVwE1Ulm0lSVIJNX1m45xzzmH06NHMnj2b66+//l3fT2taMgXo1q0bK1eubD4+4IADuOSSS7jmmmu444473tP+s88+y+WXX86vf/1rHn/8cY488sg2v81W/cmP6jLXXXcdX//613nhhRdoaGjglVdeWbeb7iD1DOKmATtFxPYR8QHgJGDSGrZxMvDj6oRido6ojPSxwBMd0FdJkrQefPzjH2fChAkA3Hbbbey7774ALF68mH79+gFwyy23tNvOvHnzeOqpp5rPZ82axYc//GEANt10U5YsWQLA3//+dzbZZBN69+7Nyy+/zC9+8YvmOtXlAP75n/+ZuXPn8s4773DnnXc2pz/99NMMHz6ccePG0adPH1544YW1vf0OVbdttzJzZUSMprIU2g24MTPnRMQ4YHpmToqIYcCdwIeAoyPikszcHSAiBlCZyftNi6Zvi4i+VJZrZwFn1+seJEnS2lu2bBn9+/dvPj///PO5+uqrOfPMM7nsssvo27cvN910E1B5geGEE06gX79+jBgxgmeffXa1bS9dupRzzjmH119/ne7du7Pjjjsyfvx4AM466yyOOOIItt56a6ZOncrgwYPZfffd2WGHHdhnn32a22hZ7tJLL+Woo45i2223ZY899mh+Pm7MmDE89dRTZCYHH3wwe+65Z0cP1Vqp24sNXcnQoUNz+vTpbeb7YoMkSRUd+WKD2tdVX2yQJElSnRjESZIklZBBnCRJUgkZxEmSJJWQQZwkSVIJGcRJkiSVUN2+EydJksqvoz/DVetntf72t79x3nnnMW3aNHr06MGAAQM49thjmTRpEpMnT+7QPq2JsWPH0qtXL7785S+3WebPf/4zJ510EhHBxIkT+fnPf861117LkCFDuO222zqsL87ESZKkLiUzOe644zjggAN4+umnefLJJ/nmN7/Jyy+/vE7tNm3XVW933XUXxxxzDI899hgf+chH+N73vseUKVM6NIADgzhJktTFTJ06lY033pizz/7HpkwNDQ3st99+LF26lOOPP56PfvSjfOYzn6Fp04Jx48YxbNgw9thjD84666zm9AMOOICvfvWr7L///nz729/m6aefZsSIEQwbNoyLLrqIXr16NV/jsssuY9iwYQwaNIiLL764Of0b3/gGu+yyC4cccgjz5s1rTp81axYjRoxg0KBBHHfccbz22mtMmTKFq666ihtuuIEDDzyQs88+m2eeeYZRo0Zx5ZVXdug4uZwqSZK6lCeeeIK99tqr1bzHHnuMOXPmsM0227DPPvvwu9/9jn333ZfRo0dz0UUXAXDqqacyefJkjj76aABef/11fvObyi6eRx11FOeeey4nn3wy1113XXO7999/P0899RR//OMfyUxGjRrFQw89xCabbMKECRN47LHHWLlyJUOGDGnu22mnncZ3vvMd9t9/fy666CIuueQSrrrqKs4+++x3Lbnee++9TJ06lT59+nToODkTJ0mSSmPvvfemf//+bLTRRjQ0NLBgwQKgMns3fPhwBg4cyAMPPMCcOXOa65x44onNx4888ggnnHACAKecckpz+v3338/999/P4MGDGTJkCH/+85956qmnePjhhznuuOP4p3/6JzbbbDNGjRoFwOLFi3n99dfZf//9ATj99NN56KGH6n377+JMnCRJ6lJ23313Jk6c2Gpejx49mo+7devGypUrWb58OV/60peYPn062267LWPHjmX58uXN5TbZZJN2r5mZXHjhhXzhC194V/pVV11FRKzlndSXM3GSJKlLOeigg1ixYgXf//73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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = df_rating_status_visual, x = 'ProsperRating (numeric)', y = 'Proportion', hue = 'LoanStatus');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> A similar trend can be seen here, ie a higher credit grade or Prosper rating implies a lower amount of chargedoff and defaulted loans.\n",
    "> This shows that the rating allocated by Prosper is quite useful in determining the likelihood of loan default. \n",
    "\n",
    "### 3) Prosper Score vs Prosper Rating\n",
    "> We can guess that there is high correlation between Prosper Score and Prosper Rating but we can double-check this. Credit Grade might have had a high correlation with Prosper Score as well, however Prosper Score has been introduced as a measure when Prosper Rating was introduced.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_score_rating = df_prosperRating[df_prosperRating.ProsperScore.notnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.\n",
      "  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlsAAAFACAYAAACLPLm0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzsvUmMJFma3/d7my3uHlsulbX3Mj09K6anRi0IBCleBoRESBzNDHQgdZIuoxMhHaWDzjrzOhedJF0IdA8hESQgEBwdhmyom1XTPdN7ddeWVZWVS0T4ZstbPh2ee+RSW2Z1uEVklf2AQCxu4WbPzN3e37/l/5SIMDIyMjIyMjIyshv0RR/AyMjIyMjIyMjnmVFsjYyMjIyMjIzskFFsjYyMjIyMjIzskFFsjYyMjIyMjIzskFFsjYyMjIyMjIzskFFsjYyMjIyMjIzskFFsjYyMjIyMjIzskFFsjYyMjIyMjIzskFFsjYyMjIyMjIzsEHvRB/Ag165dky9/+csXfRgjIyMjIyMjI5/K9773vTsicv3TtrtUYuvLX/4y3/3udy/6MEZGRkZGRkZGPhWl1JuPs92YRhwZGRkZGRkZ2SGj2BoZGRkZGRkZ2SGj2BoZGRkZGRkZ2SGj2BoZGRkZGRkZ2SGj2BoZGRkZGRkZ2SGj2BoZGRkZGRkZ2SGj2BoZGRkZGRkZ2SGj2BoZGRkZGRkZ2SGXytR0ZOSLhIiQkiCAArRWKKUGf46RT2d7npMIKaZ8vpXCaIUx+qFz/lHXBCDGREyCiOS/G41W6kKumYgQYyLEREqCUuRjMRpJiRASSSkMUBQGY8xjPY/WCmv0h87JUKTNsYeUiCGSRPAh0IeIKEWhNVVhUSgCgu8DXejpfCLGhCIhAj7l73VZcDCp2JtUFIW7sPdWjJG287Q+klLCWU3pHEYrUhL6GFk3LevOk1DUzjCrC5x1xM05QStUSogCHxJd35NQFNYyqxyTuvzY67wLRIS+95wsV7xz+x5v3znmZLEgBE/hHEh+r7Qh4JtANFA7y9F0Sl07UlAEDaHtSUZQSdEnj8Gyv7/Hy1cPeen6Ffb3poOO6+MYxdbIyAUgIoSYJzml1Nnv1vCpE/f28ZQSvU+wmSiVghT50HNcJJ8HMbi9NpC/h5gnARBiAmcUdWkxxnzomoDQeyGmLLSUgr6PBElYY6isxjlL4cxg50VE6PpA03k6nwgxfzdaYQ2EpHBGUxaWGBN+mZgWBucczmicM2itSSnR9RG/FWwaJCpMTBTWDDomyO+HposkiTRdYt013Fu0zJsOCQlrNKh8zapCgcDJsuNkvQaV6LvISbMCgaIq2Xcls8mE/brkysGEF67ssT+tsXbYccUYOV31rJqOpg+cLNcsmwYNGKuxSlj3kXuLDtFwUDi8CEop9iqL1gaFoouRpusQEiEJ3icmZc3Vgwl7VcUzR4kre/UgwkREaNqetz64x2u/fJsf/OI2b70D761hBfR4EiBkkeLJ948JAccpBpgAFTAHFkDaPPdV4Ks35rz0/Hv89ksv8B997SVuXN2/cME1iq2RkQtgG03Y3rTVZmJOSTAm/+2TBBlA1+dP7moz8SmVJ8uUOHuOoXlQXCFCkvsC6+ME5WUnpY3QCok+REKMND5htaJ0lhAT6zYyqcAHyUJrc018EJBIHxMINH2kDwGFwlrBB0WVBK3AuWFuxyFElm3PqulZtZ57qxaJCVdYjEpYVzIrDPMmYI2i85EQDYfGEVOk84nCKdou0MdITPkcWWtw1iBBQMjizQ43wXkf8TEwX655/2TFB6dLTtcrVNIkEbqUcEoRYgAl+Aghdqy6xGrd0gToG+gFpmVLNYWZNuzVJdO64ubRAV9/7grPXjlkUpeDvYa7LrBcr3nvZM3pouG46ejahnXf0zYrbq0CJsG0zq+5ZcoipCgVhQWnJ1TOsGoa5t7TdfkD2TOH++xVEKLQzBJGC5PCMp3s/prFmDherPjpzQ/43t/e5tW34XWyuPokTh7juW8BP7wF37iVaNZvczir2J9UzGb1r37gvwKj2Bq5tHweoiIfxzYN9SBbQbLlkwSZiBCSnKVrBIibSVtdoNB6UByGmM7E1qPHf1Fi8LMQYqTrE33MUZzeB5IoolYECTilqEtF34MyGq1zKWy+lIl152m9UDhN7z1d3IjhCEpB20ecVoOJrc4Hmi7S+MhJF0hB4aMmpIBSmir1dN5SO42PkERxZ95TOENhHMbA6Sqw7hOgUEoIEXQQJiU5gpSEENOgYqsLgZNly1sfLDhZNdw5XXHr9A4nq4Q2UDlFrQ1zH5AAVWVoQiREWC+h9/l9OZ/DsoZDgTtdpHRrnrsqKOXQGAKGL11TVFU5yLia3vPuyZp7x0vurFd8cHLM7UVPu4YuQN+RX3s60XdgBZIFRHAFXD1a0XdwvAAVoWkAgevPznn22pyXr19lUhScNIFrfWA62f2YYhLuLtb89I13+Ou34Q0+XWg9KT8BzM/h5evv8vUXn2XGKLZGRj7E46bZnlYUeYyPpgwfHNknCbKtsIoxgVIo8sQd4iZdcgE8Kg5RCq14SFw9KCifBjG9TU2hco1V6DyLNlI6Q6F0jlZFjwKSE2qjz65rTIk+Qu8T1mp6nzhpPEZpJsaQEIyxOfUY4mBj8iGBErqYICqKsiKsW0Q0IoFVA6VTrEOi8ZFJ5VBE3ruzxDqDM1lI79UVCWG+7FDGMHGG3uearbSp5RqStuu5M1/TdT0nyyXvnhzz87cS3RrKCZSFYGxAIgSBvRRZLkErODkBURA7aAOcNlmULBvYnwCqoS4N07Lk3nzFlWk5mNha9y2378057Tzv35vz5t2ekzswX+b0WqFgKTmNtiTfN6oOCqBdw9snebt+83ynQA2kW+AjrP1dtFJ8zV0lpmGumYhw+96cH92EN4Gwg320wO0Ir3+womm7HezhyRjF1sil5HHSbE8zWit8SIhkscRmgnb2vlD6JEEmmxRdkjxZCBBCwuj7kaSheVQcKsiT7gOCaluI/bSI6RASxigETfAen4S27+g6xf60QolGJ+hVIEd5FEZrjFH0PmRho3NqNxfVJXoSqteUVpNcyheR4castUISpCgoA33XIwjB91SlY9k0iFKQhBgj8xBxTpGiMKkc2oBWGtVHNJoEaIE2JCAyqd2mXm1YfEg0XWDZ9bw1P+HdW56797IAUW2u8emAAwezCSw1rFY5knVMFiNp830NuAZKIK2hOoHb0yVXZ/usVoF1twt58NG0bWDlO24fn/LLWw1vvZOFxF3AAEgWUA8ekQGm5HqmLfaBbTrg+wK/cxvqEn556w7PHk2xQ907RDhu53xwNx/LrrgNhA4iwwr/j2IUWyOXksdJs32ekU0njo85gmXM/dSUMepsarZGIZJFDQhOX0wXGHxYHOZi8IRW9yM9kmBSGdJGY152MZ0AawwhRnzM6rYuCladZ917KmMIKtF4mBaGPkLtFNo4QkxobXAp0SVBkbv1ui5Qu4Qz5uxDxYMie9eU1qBUriVLfaDrI0jI1y9FphMHIbAOCqcVPnb0QUiiSDHgnKMuDfPVmrpwKMB7T9QwK0o0gjVm8OuotKJr17x7eo/jk567d7PYuEe+nxxttms9HJ/CMxFMynVA9zaPbTJsQBYmnlxwHQOsGjhpVhzUFnXuSa+Px6dI33rmqzWLeR7TB9wXKRo+JCUiDwst+HD0KAHvA1eXcGUPCDJYVFwA36azaNuuaAFrQaWLv6fs7MwqpX5DKfXaA19zpdT/uKv9jXy+2E7cD/Jomu1p5qMid2qTcjvrfnsg0uVDAhGsydETbTTW6LP027ZY2wxYI/MoWmfht71uKQn53p2FVlbQQttF4qag/1HrgHTJxPRG4hKiUFhNUVq0gv26wGlovGfdBhBBm1wsf3fV0XQ9ioTViYgibVJqe5VlWhlKp9Fa4TSUVuMG7JRyzmCNpi4s9aTicOYoywLoCfSU2qALw9Geo7CaVe8BTaEti66n6TraLtK1iaZLINm+YlI6rDE0fSSEwNABViPCcdOyanraFZz0OUqlyFEFDTjyy9AB3RJO11mYHAAvAy8AM3JEqyBHh3KzR36iVdcREkxKN9i4JCU6SYhS1NNc/P6gcPpVYjYWMAoKDfnNOlRHLIgSmgH2lQKIGi5N/3HsLLIlIj8Bfh9AKWWAm8C3drW/kc8XWquzdvvtpLyN6nweyOk1HkqjpW0+5pGoj7XmTGhu/6Y3nYciajMPqLMU3UVxvxtSzgSX0YqYhI3Nz5k9gvd5ikiizsaaUoIE1siFRecexRhF1yVCjLnDMCScNTijab0QukBhLcZa+pDoQyLFwFpta6PAKUNZOkTAOse+sRRWUbgip32VRg9aZ6eYlg5jDHs+0gXDfNWBmjAtC2Z1xcm6JYTIou8ojcJpTZ8ilS1QwLxZMXMFfecxuqQuLSEEvMuWDyK5HsimdNYwsGuSktzM0HWcHufIz4ossraiqSULrQVZiBWbxxVZtFxX+S242jxnRU7JpQAqwFFRcnWvpnLDiS1rLBNnSSK0yzyuAn5loVKSa7fcBKraoSU3gwyB1ookgdsD7Ovuae7ovGiGSiP+IfC6iLw50P5GnnIenbgVm/TZJ0zCT0PB9RZJWYwopR8yYrQKisJ+yBPm0RSq1ooUQWseEqMXVa+1RSl1XxCL0PRZVG2Py4eEMwqlwfuEsRql9GZsCmO4NKnEXBenKJ0mRU0bIoUz7E0KBGiO+3zeSfQ+0hMJMeFTogmeaeEgCd4I0kWqUlPnojRiEpzVGKXQmvNvxfqkcQHWWqYWokvIGsoicqA0giJu0oApJazWlK5i2Xr61DNxk009l0I5h8QESpOiImmFRmdfJ63ROte8FcUwYqvpAnrTtFBO4aqHVch1O5H70ap28/vWx6nY/H8LlNm5IwsssjCbGZjtQTWBZw5nXD/cGzSCXJUOq4W+hWadxVZ7Ds/ryWOsDMzKCXVZEQZqajBa0TbdIJGteydwuh5iT5/MUGLrHwP/50c9oJT6M+DPAF5++eWBDmfkaeChiftTeFoKrrcorSBlw8LOC0oJWhuERNcnyiJHgR5MsykAZ87sHp5UjF4ECogiGK3PxKJInswhnRXLP3j823q1ixbOW38tUBSFxQvoTUqxDxFrDLOqYN37bAUhMU/eWhFz0SFKKyQJqhA00PoASuNMLqTPxqiKIbO/ihxl7EOi6T3rPrDuBatzWrppA9pkZ/FpUWCdwznDvBFEAghYm9BJ6CXSrjtkUlJjCJIFJ5IorB7UhqTtPU4bjC0oyx5loAp5klPkaNU2glWQBZUiF8fvbX7vyCm6rdjq2UToCjisLMooDmo3WLQO8utp0fSs2nwsHeejzQ/ZdCUKVA7qqmCoNKJSinXjB9lXAlbrL0A3olKqAP4I+J8/6nER+XPgzwG++c1vXq6CjZFLzYMTcooJpRVK5ZvgZS243qJVzld0fUIpMFpv0oAGyO7O2asq+xVthaePQpK8XMeTiNELQSnKwtC0gRgjRuuzgn6l2ER1HhZRkou+CJELF84PpnqNMZQmsuoiMWZRZZTQiuCMQUTR+SxgVO3Yc5YogqSN0EyaIIoYE6UDo02uW0NhjWTfgYHYnnMfIl2IWYAEzyJmEdx1HcH7TW1gol1HnFGUxm580yxWOU77gEWwpUNCYiUG10Wmld7UNwX2q+F6sJRSoA2agCTo+iyqrpMjVO+ShRXkKFdFfnzG/VTjcvO3CqhtTh8WFTgFB/szUoB1l7g6YCgypEQkZEuHEuw56QYB9lw2OPVk4W/tQK9DlT+QVJxPlO6T8OQmg4tmiHfCPwT+g4jcGmBfI18QHo1kJdjUOj3YDXd5uxe3aUCUotjUZG3TgFkwRiQJKaWH1t97MOpzqYUWm8/ISlFXlt4nZCN+FRCjwlmVXQ8eqcvbWkRcdKfiNtUrkt3J2z6wbFqSJIxx9CGgtaVy2Y1cRGNs7kY0Gro+u8wrEVAxO/trQalcHI+A3tTdqQHTv9s1Ha1WFMYQCPRJaJuAtZH5quG0a5kWBYd1CUrR9j0JTQievVlF7wVNoixKhEDTJ5wR+sKhNtdzyNQo5GWTkgR6n2ga8JLrmhbkyNUh2SJhQo5e3eP+MjBXyH5aan0/8ucDXL0CVQWqAJU0fYysfD+oh9i6j1jjsBaqKex1cOccnjdHWsF7UDEQBKwaqL5us/LAEC+RCqiL4WrsPo4hxNY/4WNSiCMjn5UHu/lkE0GIkpCkcJs12S5z9+I2DRg8Z1GfbRotpZSjPkZj1cORn61b/OWUkA+zbXJQSlE4nS0GyB2WuUA2i5JtB+M2lZijSRdv+6G0IoVE5xNKCasu0kXo+oBI9lqqy4irCqzRlBOHUZpF75GYhZbRCkFhyBE9rbIpqDWbyFbKka2hGxu21ioKYd55dEyUleXWvXs0IVCbnHDrk8YiKG04nNQEySLL+8j+pEREoV1Ju2pxE4vE7OReGMekdlk1D0TuYFZom33NTskWCWnzVXJ/wrPkVOGcHPW6uvm7Bg4Os/+W93B6DPZGXnN06SP7QN97+pgG8yNPSXDkdJ+POQJ3HvTk2rWug3nToUKPGig9GkIulajYrc8WwN4BzKpqx3v5dHYqtpRSE+AfAP/9Lvcz8sVjO1mcRbi0QiVFlISKgtHbguvLKrfy8Zelpd2kErdCKyUoCp2XexF5yHNsm2ZTl1ZG3ufBurKzgnCjHxpL7jDVZ9G6mOQsJfxgXcxFCGetFJKyUWznI23fEQJ50WZjmdUVjfecNh3TugClsIXi6rTAi+D7RFGYzeLgOotkSeTqO6GwOWUsadjGhrwIcMfdxZrjZc/JYk1MYLXD2oJ9U9CLbLoLI85aSMKsrkgp0seA0QbvPcoYtAh1bUg+MDmcslfZ7L+l1O68hT6CpDQTq0lp40dHjmgtyBGuHrjG/VSiJtcs7W9+XjdgFcxP8v8uyQJHbsNXXoJCG4x2rJqI98N1t9WFYZ0CbQehzWam58GCLFAPjvJ79c66zR3BA5CUop5UlDtPIkJpoRyoSeOT2KnYEpE19z80jIycG1sfrvsRrmztJylPZJLAuYsz+HxctNZURf6kl1JCk4WW3hSUq5Q717a6I3tXZbuEGNOl77zc1pV9kkntoylhpRU+CM6ms/NwEbYfWisSuVuw87kzsY+JFHPTQjWzGK+JErFKMyldjlBaQ6UEVetsexGziARBRJFCpEMorcFoPfjrtOt67ix7QtQYky0g7q1XFNZjEaIxSNfjioLSOKJAXRRoIzhjEQWzUnNvpSmMoihLVNT0wNG0xOhsU0BSg3UiAmgUTcyLYF+5DosGQrhfGD8hR7EOyeJqG93a2zx+cAg6wMkKupQjSD1gXN5Y6VwvqQ3IgLHl0hlKbSmKHIU6TzmkgbLUVNUEiUIYqLZJi0DIEbtdU9SbBeIvmNFBfuSpZJuiSiJnEzKos4nr0WVuLjNa64+clLbL9+iYNpN1rkvRWn3Io+syd17Ch93l4X606lGDV601ziYkCaIurtNSKUVhFKu2R1TaOPcrtDZEwPuAc4rSFljrKAtLufFDQvIi4ZIiWudW95zqVlir0SjcpgVx6BTiqvNYpSjrEi8tk7pm1idS8lR1yb1FgzGWyhZohF5SNqyNMJs5ZliaPnA0NWido3NFXTFzepMejZTGUhZ60K69urQoSQgKrHD1GQjv5oL37URXsRFQQK1hOoGqzmlCbfKGezaLMi/QNnD1MD+mEURp9qoSa4acOg3P7B/wzOyE4324Mc/jucNnT8HNyI0D0xk4A4VV7M8miAxzvazV6FKfW0r04/gKcDDbLiV1sYxia+SpZJuikpQXC9abCMplr9V6UpRSWGseeqPGmB4SJ5e98xI+2aQ2pg8LY601ooZbPuTjcM4QW0XtLCEKum0wRuN0TlnXxlI5Tek2Dv5GI1GIKWBR2UIBgzWKEHIkUmu1EZB6Yy8xLCFmn60ETJxD0FwRYbFqcEqzNykotcIWjpQCB2WFs4ZJqZnWJVaDsxGtEj7l1HddOAqX69Aqa6hKO7g4nlYV+7OaWWlZeE+h4fp1KI/zfWJWgq2BTadiNYX9WbZ1EPKizZJgfw+0ZbPgtmAL2C9K6rLkxv6EaeUoB/TqUEZz/XCPq88c8rKccDKF6RL2F7nQf2vWuiRHqrZmrgfk6NzWziJufj/a/H51As88A047ZvWEZ/bqwZaN0sZwbe+Qr1xrWN3JjufnzT7w4h7szQrKL0iB/MjITlAqF8M/6q/1eXKa/yiexnUjP8kXTCEfG/W6aLTW1Fbjk1A7A7OCdRdJSSitPbuJl86epXIVEaNyt6XyirCpsbObiWy7uLVi2DURt5RWsWgCEY21hioJa5XYmzr2JiWxT4g1OK2pnaVwubi/LrJ1h9Wag4kmSba6gLTxUUu4jd3HRURYq8Lx0pV9ls2zqNt3ObZrll2u2dEKjvbg+uGEmCJdzM7/nQ+kEKnKgupI00tiseqxxlAVE7TTtL3n+l7JtdmM60czqrKkLIebOitrqOuK52Z7ONH8RO7hCjicwXoFScPeBO7eySJS2ex2X09h1UKMMHG589AYuHYddF5liaMKrs1KXrqyx3Q2pS6GGZcCnjk44Hd//ZggLfYuvHmOzz8FbpDP0Y3DKdf3Juf47J+NUWyNPNU8Leae58knpeQuMx/nC3bZl2YqC0vqE5NKM6kK9kOg6QKFVTijEVFUG4ERY+64nNZ5FQBnDT6kbE67eY0Wzpz5pF3EdZvWBSsvRO+RZNEqsVeX7JcWZzV9JZs6wM36m0oxLS1lYc/S15BXAyidwmh7tn5g4YZNHT6Ic4YrBzN+LSamZcEHJwvmXUvwnsoa9uopRZn94jV5HK0PRIS9oqJyBlLizrqh9ZHKFXS+J5aWZ/dnvHhtxpVpxcGk/NAKD7tkUhUcTUquH0xIaL4SPMu2RRvNuu1IPdQTxUtHwiJsVwiArs21Z7WGegKLFfQJru1pppOCUhsmZcWXrh/ypWcOuDKrKAaKAGmteOHaPrdOnmG1eoeiSEzfg5+QI3AfhfmExx7c5go5RfqV5+BrL1d8/fnnuHYwO8ej/2yMYmvkqefSm3ueM5ddnDwpl10wW2soBUKI+CQ54nOWclFoJcQo2Znc6FyPou+b6zqb04Va5+0edsof/ro553hmH5ZrWPWJaWm4cVDjnCVtvOpiSCRyPZmzOQKWF0u/byTsjAJUdsrn4ps0tNYczSpiEgrnuH54QIyRKNnRXtDMase0sHR9ZNl7nALnLHVh0crQhMCNtqcJntWqp5eK67OSG1ePNiLLUpXDLvbunOXq/gSjrlJVK45qw0nnsSLsVTWKyMInamsJsedk3WdbmSRUTjEr6+yOH3ta7/P6o0XJUVVy9co+L1zZZ1JXD3UK7xqlFDeODvitLwl7heLalWNuHMz5tQXEJkfhrMnRuRAByfV1McDxCZwsNmlRB/v72ezVFbBX5rRw5RQv37jGb714g5dvXKeuL976QV2m1MM3v/lN+e53v3vRhzEycum5DMvZfJHYnu+08XRTWqE/wgH/cZ/nsly3y3Y850GMkbbzND4QfMQ6TWksRmcT3QSbNG8WxUmyuEQrSImY0tlKDUapsw5Tqx8W0kOyXbar6z2tD/R9IEhAY3FO5eWFRBMkIjFt1qbM1hvGWqzWOKNRelM7qBRWKYrCDBqlexTvPaumZ9V62tAS+rxGbOc72i5mwc+mQUPy8k/VZqmvRduz7noUhmsHE65Ma4qqQkluOKqLgmJTR7nL17RS6nsi8s1P226MbI2MPIV80aJ5F832fP+q09Jlu26X7XjOA2MM04l5sk63i6+f/kTOGmXsE47royjP44jOB+cch85xuH/RR7J7Lt7pa2RkZGRkZGTkc8wotkZGRkZGRkZGdsgotkZGRkZGRkZGdsgotkZGRkZGRkZGdsgotkZGRkZGRkZGdsgotkZGRkZGRkZGdsgotkZGRkZGRkZGdsjoszUyMjIyMjIy8gS0PvJvfvzBY28/iq2RkZGRkZGRkU8hJeHf//Iu3371Jv/yB++z7MJj/+8otkZGRkZGRkZGPoYfvTfn26/e5Nuv3eTWvDv7u9GPv/rCKLZGRkZGRkZGRh7g5knDv3jtXb716jv89Nbyoce+8dIBf/rKi/wXv/cc1//Xx3u+UWyNjIyMjIyMfOE5XXv+5d+8x7dfvcl3fnnvoce+dHXCn77yAv/V77/Al689+QqVo9gaGRkZGRkZ+ULS+si//ckHfOvVm/ybH3+Aj3L22JVpwR/93vP8yR+8wO+9eIBSn33R9lFsjYyMjIyMjHxhSEn4zi/v8Rev3eT//sF7LNr7he6V0/xnv/Msf/LKC/y9r13DmvNxyBrF1sjIyMjIyMjnnh+/P+fbr77LX7x2k/dO27O/G6X4u79+jT995QX+wW/fYFqevzQaxdZjICKkJAigAK3VrxROHNkNKSVCSCSyW6+1Gq0vl2/vR72WgCd6fT04TiWSt9f6Ql+b22OKIqQQSQgiCqMVhTNYawCIMRGTICJIyj9vr5fRCpTanAtBoTBGY43GGH1h4/I+0oeIDwERUApiCLQh0vmIUkLlHKWzGK1RWiMpkURIApDH2vWBHsGKZjZxTOsKZ+2FXLOUEl3nWbYdq7YnxoQxoAQiIAmc1Tij8SEwX3WsvIeUcEZhjMUYQ2kNVeE210djjaEqLFXhsNYMPq4YI6um5XjeMG9b2ralC56+i/gY0EZTGouofO26XigKxX5dUzlL03cs1hFRidIqJlVN6Ur2piVH0wmzSYVzdvBxhRBYrlvm65513xJ8ICVIKqGSwseeVesR0ZSF5nBa4Yxj3Xcsm54QBSUenxK+F7rkqZ3jYLbHM4czrh3MmNTV4PdLESHGhA+Rznuapmfte3ofscYwKS2FNSQUMQrWaEpnQIQuJkIQIJCS4COIJEpnmVQldeEoCsutRbcpdL/Jj99fPLT/33vxgD995QX+y288z7VZudOxfmHF1uMKKBHJL1QFSqmz361hFFwXwMddt5QSbZ/QGrTWZ79XBWc3kIsWzSKCD4kQAk3nWbU9PkasUVTWUVYFTm+EBULnIz4mFIK6eEpMAAAgAElEQVQzBucsWoGPoJQQo9D0gRgjVWHQ2mC0oi7zRDgUKSVWjafzPYvGs2g7FIq90oI2GK2ZFln4BlEgCR8jTZcQIilETltP3/f5fYUmonBKcIWlsiUHs4KjWU1RFIOOa7nuuTef886dBcerjsIkkMRJ62l6j+89Eag0GGsorcVZRwodS5+3bX1HDEI9mXF9WiNaESPs7034tet7XDs8YFKXg70WU0osVh0379zjjVvHvHfvhFvzU1ZzaDwYA4WFugBroe2hTxA6WHcQE1QFTKYwseAMzOoCW1U8szfj+uEBL13b5/rhPlXpBhtXjJH37p7w3R+/znd/fJtfvAUnARogAB35Z02e+Nzm+2LzGMAMmG626YBD4Pk9uPEsfOW5Ga985UW+/tJzzKb1YOMKIfDmrbt8/5dv86Of3+KHb8Bxm0WxAXpgG6M5eOB3T77PGaDajCc9ML4jC8+9AF95ruLrLz7LN778PNevHAwmuESEtvOcLFfcOllz93TF8XLBugskhOA7uig4o5jVjso4QjJoegKGvdLRRuF0tSSh2XOGdUwoEbAF75wovvfemtfeWSAP7PflKxP+5JUX+ONXXuArn6HQ/bPyhRRbTyKgUrq/HWy/50nbmKdbbF20+HhSHrxuACEmUgBnFDEmtFZnN4r8PUdbnMuP+yhoBcZoBAYXzTEm2q7neNUxX/d0PtD2Hh8SexPHXl1RFhoiGKcprUOUou8TSgXqmEVKjAlBo1U+B4u2R601V2YlVmtSgtlEDXbT7DrPsvM0XeBk1XG6WHFn1eJU4nA6obSaVR+oK8u12RQfhWXb0feRVduyDj3JC3ebNd57DidTKlfQC1ypa545MiSgi4rnDhXOucHG9d7dY37w1m3mTYtCcffkHrcWARXBB2g78D00DVQT2N+D6OHeHLQHXUDTg9bwzNWOapZvutf3JxwtGxZNy9d6+OqNA+q6GmRcfR+4efsOf/m3r/OLd1e8ewveXcMdYA0UwIQ8MS/IxzslT+7buMDh5ms72Vf0PL/fc+35Ob95Y8Wy8SilePbKAc4NM83Mlyv+6m9+xr/+zl1eP4U3PsNztOTzsOVd4IcLYAEv/mzJj372Y/7o77X8J7/zNapqt5GQLXdPTvnLH/yMf/f9U1699/DxPcrbT/LEAXgTfvPNlle++gYhRv7TqmBvNowA8T5w63jOz9+9y93Fkrfu3OWdW54AOAdNm8V8FFAa9qfgFLz3AfQCe1UW/m0HyzV0DZxqOFFwHPPrd8vRxPGPvvE8f/LKC/z+S4cXMs99IcXWkwgo4cOT8VagPc08jRG77XUD8CHlVBTQ+0RIQlk8/HLWWhNjRMV8bbcpuzxOhVIMKpp9iCxaT9P1rPtI7xN9I/jkWXUNd0/XlM6QdGRmS64c7eF0Ts90PtD1QoiJLkY0Bq2ERefRWmMUnK47qqLIIiwYimIYsdWGiEhi1fbcPl5zr2lYNS2rZs37pyuqwmBEYbTmpllgjEYrRR9abq3WzNcthYJF61msoHQrbhwVHNT7nGqNPlHI0ZTCKNat5WAgsbVoWt64fcqi6QiiubeY89M3Au/fgxV5rmoe+LrWgrmX7xmBLE66LkcbLPDeOznacAR8+cU1L96IRGCvLDmauMHE1rJt+eu33uf191e89z78rIEHFx3ZjmdL4H7kZMvdzdeD3JrD83M4vXPKf/zbiYNpwdHedDCx9d7dOf/u+3f5y9N8zOfNO0D/Hky//wZfunGFL7/43A728mFev3mbv33zlL/+FKH1WfkxsPgFTCdv87Xnrg4mtpZNyy/fO+b90xW3T0752594frGAJTnC6MhivgQmBfyiz++fBqjJ78El+bW5jdo9qLA08NUj+KPffZ7/5u9+jWuHe4OM6+P4QoqtJxFQiixMHtxeRLiccuTxeRojdtur0/eBPgpGq7MolUjC+0BR3J+IU0qQBGUB9WDUbjtOPaho9iHSh0QbhLaNCDD3DeuuQzAcFBoRy6oLLIuEaIMyOTVllND2gd4Hui5gneFkOWfeRa5PZxwdzQjR0PhIoYVJPcynboAUhaaPHC9W3F2vWDYtTUg0PnKynrNuAkUBB3W5iaBCG4Xlqkd0ToveuQvaQF1BewLzpufZGwteEMXUFjRNYFUWTH3kYKBxLdue1bqji4n37x7z9t3Iu/dyxORR8QEfPxE+uu0HQHwHutihXhAOpjWnbc/z53nwn8Bi3fHurQ+4fQdea/KkdR4cb77euQPVTxZcP1jya8/2zCbDiMhfvn+L776/G6G15QPgRz+D179xbzCx9ebdY95862FBfN7cBH7yc3j7N074+ssv7HBP9zldtdxbrZkvV/z4rYYfLO4L+HuPbHvQZ0HVAY9zxz4AvmTg2RKem3j6kD71f3bNF1JsPYmA0loRogByJshEuLSC5HF5KiN2IvgoBMlCCXKEy5lchN32EZvSWc1WSrlIHiCGSEj5GhqtcspxYNGcRHJ906olRJAkzNcNJ6sle/WEpUpMlUJpaNYr7oiiMJo+BiRF+hgpnWXdJ6IkVp1HK+Gk6XGug6lihqVRkasDjsto4XTZcOdkQdcFFuuWW6cLUoJ1C6slFBUc644+bmpmDNw7zvVBMcJ8DWUB6QD6HtISLB318ytm05K6d4SmRabD3bJSEta+5Y137/DTt+D9LqeVflU88Drg3oPK9ByU90gvDHfF2s7z1l345en5Ca0HOQX+7S34+q13aL46jCABeOvO7SdLo31GfpLg9r3TAfaUmc9P+eu4+/38vIWT9eLTNzwnWh84Pp3zk7fnvPP+hyOlD/KkZ3tN/hB3egrv3rtLDAOcwE/hCym2nkRAKaWwZtMttpmcjbnctU2Pw9MSsdt2q8QkhBBJsolYKTYRLcEotUmlCcEHRCmsUpSFQQR6H4kC29H1IWE0aKVwdrjuGwVoBUkLTdfTxMC6a2l9QNuePgSsNqSYWPSRIGvKwnG8XNG0a5KAtQpnCoIkfO8pXUmIS2qrmNSOLsimU2zg7jYgSKJLER89zQo+uAWtz2H/qYGyBnpYedACswpWXa6vMGSRtbydi3mVgeUC7p6seWZvj8Y2dBQUdrhbljOwXi25eRdud+cbWejJUSB9E567uqIaKOUL0IUW8R+OHpwnp8DP3k2suo+KAe4G3w8zoc6B+Xo4sbVYR/wA+3kPWK2bT93uvIi+5/Zizq3bD6etzwNPFm97CVZdYsDbxsdyCQ5heJ5UQCmlnvpI1qN8kuC8LIXz2+69uKm3SkDrPcumRyC39lrDOkJpBaPNJpKlMBqSKFIMtH0gbVr2NyE9rFJoNWx9Wq4Z0+yVBV0RWCxaEolCB6wIVWlZtS1JEin0NBJYtA3Re3wMtEHoG09VJqzW9H1gHTxXqgnLvqWel6SZ5cZByZABypDgaFJwtFdzZ76k9x1Nc19otYCKsH5gebE50LX5BnSbXJexLcTec2BVjngVCtZtw9GkoJB0bgaDj4Mzlj4mzDoLrfNOT71NFp1K8r6GwgfQNte77JI792DZdJ++4Tmxng+2K6IfLi21GnBcMuBtvo+BQL4l7+K1eAI4DQd7JUYP1539cez0zqWUOlRK/XOl1I+VUj9SSv2dXe7vScgC6mI9fHbJNiIUYsrda4/MvllwqvsRLsCa+wXk2zTjtmvvItKLWzGsdRaAnU90IaGNQaFp+sSy7YkhICJnvlpKgUiu41r3KV9bpYkJoigKq0Gbx8r9nyfZNwrKwjGtKvaqmuuzfQ5n+5SFpTaWddty0jZUVU1V1BhrCMYhKIxxILBYN7R9i1KKECN9SjRdTxdbrNFobfEDh829QO1qrhwcoEXoY76JOrKQWpPrLY7JN1YhRz9ubf6/JUd7HDCrAQ3VFA4mFVVVY3B0KFIcbpITBV2MRJO783ZBAJoOfBzueoXwSKvWjkgpd4sNRT/gvlb9cPs6Hi44iNPDif7eKyqtsPHJ04SPQ0kWWy8e7KN2K3Uei12f2X8G/CsR+a+VUtuO4pEd87idhh8VsYsxXZrCeckHgFIK7yNag4hCRFEUGpFE23mcNcSUHjpmESGGREq5VktQOGtQQEyCM4KkYeWWUiqnPJUmqXxjm9UOo8DHwKrviHgcwrpd0QcIMWIweC+Iytemb3Nq9HBmICTWPlIVEVGa/arAJ+h9oK6G8aRyRtH7QIqRwhh0WXB02CMJjuf5E50jR7Ma8k3QkLv1tqXTQu4wqoDTRfZwqgvoYmDZNBxUJZUGP+A163tP30aOV+ef5thSAE3MnVmDoSJNl6/LLvXJtICqGLLGbrBdMeStUAYKDk6BqhrmeokIkcDNufCmyh+0zpuebBsh2mDsxQdTdnZmlVL7wN8H/lsAEenZzTkdeYRfpdPwMhXOK8gu4wJ9TLnmSnKULoQcfdtGJWNMtH3AGn1WG9X4QEzkYnQRksT8uAilsyg97BtwmwpVSiiNZmU0sfdYpViGiGhNXdRMi4omBVAd8+UaZQwiCVEQAxiXU08CaKvZn9QcTiZoZeiCMElCHFCUOGspjWHVt6y7lugDMWYxBfk6NuRi7Kvk416THw9kgVUBM71xLndQupzqUkXJZFJTGUsvmjjgjNp0npDup0J3gSZP3IsB021GFyi9W6FVAvUM9gcS/ABpyEzRgLeOsiaHhAegtrvvYr41b/mLV2/yf/z7N3ljh+OqyR5cd07nwyrxj2GXMvar5HKM/00p9Q3ge8D/ICIPNcAopf4M+DOAl19+eYeH88XhVxFMl6VwXjade51PeekWEfoQWPmAFgUkYkobp/geazUzgdIZnLW5gDzkJR8qY/Ah5vqv4KkLi9osBzMkis1yKAlEayaFZb5ccnu5yJE3k5d6SQomxoC2JImkLlJPLCHkyJZOOeqzX1YonSiMwUtOFy/WDXuTAj1gOkBrTWGFto90PmE3kcYQNmKCLKaub75v+50c+dOXBq4qcBaKAqqDvF0SqJ3BiaZLEZVSXtJnINa+B4EDDfspR+bOmwRMyhyJHAprcw3dLtkHrh6AHjAEdH022K6YlsMpu4Fsr7DkEoddsGg9/+pv3ufbr93kr16/+1BN6TbKfd5MyB/Y5m1P2w/3/vo4dnlHtsAfAP9URL6jlPpnwP8E/C8PbiQifw78OcA3v/nNS+w7cDn5yGJ2Prtg0lptDEMT2+InNXDX3jbtmQTKwhBjokuBPiQKoJfEuvN0AZzJflt72pAkW0FYI6SYsFoREpACSSkEQStDWeQaqCQfPk87HRdQFBbT9/gQCAE8Ca0NhbGEJBgUd9ZLSomgLdPCkGzFQeXwEU5WJyhgWjumZUnje9Z9R2EMdmIQUbnbckgdKcKqCTir8OIJMSEpp5G6db6RKrKZZ7P5uSffDAMbg0LJ3Yj7gOmgVTAx0LZrOltQGoWxatBuRESDgdKA2dEHYwc4p3ADLq9UaEs14ZN77X9FHHB44Fg1w01y+/sVu4tBPszedLhlXoZqVE1sOr3PiT4k/t+f3uZbr93k//nhLboHvK4OascfPF9RxgVvvgE/Ore9PrB/sgiprKH5nFs/vAO8IyLf2fz+z8lia+Sc+LjaLKO3UdOn0xvszCleKfTG1kGp7I3le8Ot+RqloLIKbQwx5JosHyNW22xwmjbNASnXOqUAWglIPh+bNY8HrUUTgbqwrLe1Y5IwaGZVjU8JCS0rHwkh0CbPbLNUBalHlGNSlNTlNWJKSAqbTkZQyqAQeu8xM0Plsuv8UMSUWIaIEgNJMV9FNu4cKA0u5ejVinwD9Gxu7ORPtUuyIDsEDgXmK5hMoKzgpAlM644X6ylWq0EjW5XTxABLv5uolgP2DEBez3IotNXsuoSlA1KAeTuclUA5kFM9QDXg9TpdD7Yr+viriWMR4T+8dcy3Xr3J//X99zhZ3zetKK3mD3/zGf7kD17k7//6Nf6/H/6cb39nwa4+ZnTAsst1yAxcn/tR7OwVIyLvK6XeVkr9hoj8BPhD4Ie72t8XkY+rzRLJnYWfxRtsu6zNg5P1Nno2mCghj0Uh9yNPSmG0ITk4nJa0veBTXnxWF8KyC+wphzjFovF0PlBYjbGOlLJlhNEwqxxGa1Aqd0pp2dmb/VH0ZgFiESitYpWdN3BKYaxh2Qht39OFRGENzhUUZH8xHxJ7M0MfNVUBKTjaGEjGURpLVVZMyhJjC8zmfA1F7yMmRlZ9y0mzpGuhDdAucw3Wilx8u60GWZKjXNslOe6Ro1uOvJHdCDXfQV1DYQx1WWKMHdTSwppsX8Hm+M67Pf0a8MwRVM7i7HCRLd97VAHPkb2VdsEx8N4dT98N4RCVaQa0Y8AP90Jc7NqjY0MB9J/Rq+znHyz5i9du8q1Xb/LO8X2BrYC/82tX+eNXXuA//91n2a/upyn3JuXZh61dUJA/xKUUKdzFBxp2Lc//KfC/bzoRfwH8dzve3xeKT6rNerTTcGsF8WneWZehQH6bBn3QCwwRFEKKGzPSQoMPeTHqKBvH+NyRGFPa+G3ljsWY2Ag3sNbio2xM7gQZ8P5sjabt8/IvMSnqylD7grurBiOJ/bJi1TU4JczqCSkJhXVY5ShLi9EanXqaNuCc5bDOnYdJNJPCsT8tmZYFbYioIZchipE2Cqd9T7vu6SSvCXgacuTKkj9l7hvOHOQ7csqwMLC/KaavdC6On1TQNrBs4cpRTnt1QbBaEQccV+eFusgRuF1MCIHsmn997whjhiskb0MkhXwNdkUCPrjHoIXkaqA3swYWabherzDQrk6BNj5+GvaDecu/+Ot3+fZrN/mbmw/Hfn/ruT3+9JUX+UffeJ5nDz56uSarNYXbXaevA27sg5vUl8Jna6diS0ReA765y318kXnc2qwnWXT6MhTIb0WWUjlCFzdmPUZrJiXEpLObPOCMJqhsP4De1mVpysJuBJhQFRarFUkEvYn+xZiX9Rkw24YxmpiEqTMsjCZ1sFfP6EPkeLliv6q4NpmiXP6+DpG2W9MlT60NguJosk8fPY5cFD8pLKW1TKoSUQqjcgpxyE7LmCIxJqbK4KqSYt6xP4G4gl5BLYCFssxpQrXKqURN7qw8NKAL2Ksh6fzzVIOtYVbvYQqXRbTkSOZQrLoeUbA/hWdX8P45Pvc14EUN5RTKylC64SaD9bpF2RxZ3JWL/JRcrzekfWEfhomiGcB3w70OXclu1lV6BAOsV58stpZd4F//zft869Wb/NXrdx7Kzj1/WPHHv/8Cf/zKC3z9xqcv+tx5Tyf5tbLP+abqy81zzg7gxmyfHVuKPhZfSAf5zwuPu+zQk1hBXIa1IB9y+CdHhLZh4Bj/f/beLMiyKzvP+/ZwpjvmXFWZhcLcaHSjgaomyCbdNMmmOKjZZKPAZ/vBdkiOsJ8cDikYsh9sK+wHOqywQ36wZDscinDIDtkWChzEoZuSmoMl0d2sAnsA0QAaaACZNeZ4hzPtwQ/7Zg0Ys4DMUwn0+SJu3My8w9k777n7rL3Wv9ZyZLPxJZHCWEdpDN4pOklEqhWxEtTW32xSraTACUGkZ+HJ2VT2W/Y0OS+lJHEcMUyCgN9ZEKKHBrRWDHtZyOrzBoWnk8YMSEgThRaKJI6paolSUfDaWUMnjYjjoNvqpJpOpGjSpSCQxEmCjmKWOj3MkmA0Lch6MB1BnkO/H7LgagPYsPBEGSQ6FA/NUohScDbcdB96sUQJRSw1nVTPxPHNfl6RCuNarUDVoWHvRyEmXAQeSWDxRAiTJiqi12CJBCth2BGc6niiaRDXHraXawAM5yFLs0N+5/dGKs3R5LXdyRKQZc001wY4eYKj7a00YxWo36WTQW0df/LydZ67uMHXvneF4rZw7SDTfOVzq5w/u8qPP7Aw65JxMCoLS13JidTBrJvEYUyzC9wHrC7CwgCW+r2bPXLvJa2x9THmoG2H7iY0eFx6Qb5XiyStFb2OQEhxU7/TcRHGeuJIBSG9FzjvEDi0AG8tSomQsaeC6D4IrcVdLQ6HQSwlubf0uylRbSlKQ6Qlw05MXjmkgCp1bO9OiLOIRCviWNFJEgSe0sGwFwOaNFbsTEq0FKRxTDcRZHFMHMlG5xXHmsVezNaupnId5ozBOUsa1cwNNZNtg07D3rKTapYXDKkO9c+UluA9Skiq2mI0SAlxJFnuDxhkHU7MDRhkKVms0Q1qm4adjIW5AYuTPVwFgzp45K4RjCZNKFHRIQj/3947cb96vibs2pcJWrVMwmA+eMzOLAw4OT+kmzVnlPTjLicWFtg6scn0tTCuaxxeEcRVgkHy6fsF873m6lgvDRd4hMu8csTHeSCFM4vNNQ7/7ANLPPziDV494uM8NA8rvQGwL3Tf4flL6/z2Cxts3yZ0j2dC9/Pn1vi5x5ZJPuR3MoljlubmWTu1ibkCcR7CzzsHfP0C4Xu4X7evT/jOLUpYXIKVBVheGrA0PyCLjqakxd3QGlsfcw7St/FuQ4PHvReklJIs1qFw56w8xX5NLiFDwdIkkjf7KgI3DTFjgvpaSXFPjMgk0ajKYIwljRTOW5TRpN2IsqwoHAwRLPRjnAtGYaIFCDnztIQ2Q0XtkDjmOxFShhBrL43I4tCyqMkeglmk6aYJ88MuSpZ0IomSHktMP0ow/YpIJRjvSGVML40onaW0llQq4kgxLmuKskBpRao1WZQSRzG9bsJcN6ObaCItP/TC/mFYGna5b3GOvekEjyWKYDCBU2U47XpdwM0W+ymsEL5r+74VSzC2OlHQplkZDKzFZViZ7zLsDHhgsc/J+U6jmXSnlrpsbGc8cqpH5cfo14MhvEUwuOYJGrU8TC+sHwd87weB1S6cPAWPnr6PpX5zxtaDJ+Z4cvUymxtHVwN0DXjyUXjgVHPG1uMPrPKTD9xg8vrhhrJv5yngkTPQ6fb4e1/7Ps9dfIs3t+4Uun/hoUV+/dwav/zESYbZRzdeVoYZC72MlYWIsqxRCuw4eKZiZs3pCUaKIHhf9ey2HyLULjStDwWeYb4PKgllY04sD3lidYmlfpc0aY2tlgY4DqHBwyaECPfDoxLvJUqFkg6IUGssnl3ArL1ldKWxuqe9MJVSzHdjdicVtfUMkgifgEfQSyOsdVTWE2uJIHjhtFbgLFPjkB60liSRwxhHEimUClqtONah3MIsXNkUSRIxdPDgYo8fCkk6Fcx1TmCMoUYx39FYF6rgoxSJEFgl6UhLYSRaeSKlglfPOIyfNaJGkkWSJInoZ5pOHBE32P5l2O3w2NoK3nl62Q6b3RGbu6AlpFG4V3GoAJ8mmth7tkrLeEIoTKuDJm2h1+FUv0+JZZwXxHHCqX6P+WGfpWGPYa9D1KBma2VuwJmTNZHSdGNNzA7Dbfi0gljDqAJyKF0I+w77oCMoaygnwfM4qYPAv2bWAUDCYBhKdgzm4ekzKzz14Cr9Bj1bD51c5qfOblOYK7x8DV4+5Pd/EvjcY/Czn3uU5bnBIb/7e3Pf8iL/9tn7EckPefH78JI/vISNFHgihmQRfv+NhH/wwp3+s0+f7PPsuTW+enaVU8PD9b4uDfs8eHKZurZE0Tad6wUrc6GwcaJCkWNfw3YdGtInMUwr6KWw2E+CXkzCIEvpacXEe6SDJM2YzyLmBkPuX+oz6CaNesTfC3Ev2rC8F08//bT/5je/ea+H8YnkXYuffsybb3+c5+Sco65D30Z8KMyqZvW3QmuhW95HqeRNYb8x7l1fA9w0Mu/F/8G5YPyVdU1eVhgXssOkBAgePO88fjbGSCkiHQzfWZx7Ng+P9Z66tjg8EkE8SwKIIoWUzRbXLcqavfGEq7tjNncnVKaiE2mSOEZHEoW6mWRhXWjFVNeGSVGHIrZaMuxmdLIMLW9tdJCSTGv6nYQ0jRudF0BRFFzfGbE5riirHFMVFHWoZC+0pxdnJLGmMiWjqcF4Tyw9SaypK8vmeEpZFZTWk2kJUhLrhEGvw5mFOdZWFuh1M1SDxVoBRqMR333tDV54fYP1yxV5AbhQr84Y6GSQZhA7MBHYCiYTKE04DZUDlYKsIZega1Aahsvw0KkeP/Ho/TyweookOfq2NrczmUx4ef0KL75xhTev71DsgdWgLRgHZnYZz2KoSqhqKEUIWXc6of3VpIKyAAzILlwv4JWx4PVdf4fQ/dQw5Zmza5w/t8qnTx6tUVmWJTd2R1zeGrM1GoFzpHFMmihiFSO1xBmDAbwVIGqM8TiviGPBME3J0gQE1DOtK0bS6UYM05RuJ0VrdaTroRDiW977D0wEbI2tlpaWlvfgbgz6d3su8LHdEByEj/OG54PGftzm9lHPRes8f/LKDZ6/uM4ffPcqeX0rmWCQar7y5CmeObvGT9yl0P1HnYMaW20YsaWlpeU9uBv94ns99+Mcrv8gjru+8/34oLEft7l9mHPRe88Lb+1y4WIQum9ObqVBRErMhO6n+bnHlkkbDGX/KNIaWy0tLS0tLZ8gXr8x4cKldS5cXOf1zTv7/XzhwQWePbfGl584xbBz74XjPyq0xlZLS0tLS8vHnM1xye/85WWeu7jOpTfvLKDwqRM9nj13mq+eXWVtrrkyIy23aI2tlpaWlpaWjyHTyvC1713lwsV1/vjlGzezrgFODBLOn1vj/Nk1Hj/VXPZky7vTGlstLS0tLS0fE4x1/Nmrmzx/cZ3f/+4Vprc1j+4lmq987hTPnFvlJx9cbIXux4jW2GppaWlpaTnGeO/59vouz82E7jfGt4TuWgl+/rEVnj23xpc+vdIK3Y8pH2hsCSFWgC8SujDkwHeAb3rfUIv1lpaWlpaWH0He2Jxy4dI6z11c57Ubd3aj/vEH5nn23Gl+5XMnmes011uz5cPxnsaWEOJLwG8QWhBdJLTPSoHzwMNCiP8b+O+894fZrLulpaWlpeVHls1xye9++zIXLq7zF2/cKXR/ZKXHs+fWeObsKqfnm6vO3/LReT/P1q8Af8N7/8bbHxBCaOBXgV8E/p8jGltLS0tLS8snnryyfLYbGrAAACAASURBVO3Fqzx/cZ1vfP966BIxY6UfhO7PnF3lM6cGH5uisS138p7Glvf+b73PYwa4cCQjamlpaWlp+YRjnef/ffUGz11c5w++c4XJ24TuX37iJM+eW+MLDy2iWqH7x56DaLb+G+A3vfc7s9/ngf/Ue/+fH/XgWlpaWlpaPil47/nO+h4XLq3zW5c2uD4ubz6mleBLj61w/uwaf+3xVuj+SeMg2Yhf9t7/nf1fvPfbQohfAVpjq6WlpaWl5QN4c2vKhYvrXLi0zqvX7xS6P33/PM9+fo1feeIU891W6P5J5SDGlhJCJN77EkAIkQHNtjxvaWlpaWn5GLE9qfidb1/m+YvrfPOH23c89shyj2c/v8ZXn1rlvoVW6P6jwEGMrf8d+CMhxP8GeODfB/7RkY6qpaWlpaXlY0ZRW77+Yqjo/i9feqfQ/atPrXL+3BqfXW2F7j9qfKCx5b3/TSHEt4G/Bgjg73rv/+DIR9bS0tLS0nLMsc7zr17d5MKldX7vO5eZlLeE7t1E8eUnTnH+7Bo/9XArdP9R5kAV5L33vwf83hGPpaWlpaWl5djjvee7G3tcuLjOb72wwbXRbUJ3KfjZx5Z59twav/D4iVbo3gK8f1HTP/Xe/7QQYkQIH958CPDe+7az5THBe49zHuc93nmEFEghkFLclat6/3084UO+29cfJvtjsc5hjcPhcdaFsQmBVpJIK4SAurbkVU1R1ngBSaRJtUIphRcCnEdpiYAwPyGQgNYSKWXjc7PWUpQ1o0nB1mTE3rjEOE8kHQioa4HQgl6kybIYrWKk8FR1yV5uKCuDVpBGCoegrj0qFgyTDgvDDv1OShRFjc7JGMPeeMrGjR2u7I2o8xqVQCYiauEo8gonJd0kYXnQIZKS3bJimlscFdI6SgdFXSOFJ4sTulnGXC9jvtejn6WkaYRSzV648jzn8uYO6zdG3NjboahKlNYIb8GBdVCY0DpFSUUaKZRWJCohiSPSWCJQGO9xlWFicsbjklpJlro9PrW2xNryImmaNjovay2bmzt8/+o1Xt+4wY29PWIFWitinTI/6LE86NHNNHlh2RkX7BVjrPHEkUZJUAhqIUmlopNFRFKB0qRxyspcxvKw3/i88jznB+uX+YtX3+SV18esb0BVQ5TC8hD6fehkUFZQVDAaQWUgVpB2oJeE52KgBGwBKoF+Dx46eYrhYI5vXS753e9c4+Vr4zuO/WNn5jl/bpWvPLnKwiEL3eu6Zm+SszutMNaSaEGiI5TW4B1lXTEpLd4LsgikgMIKvIduIkkiTVV7plV4vZKSLInppQmdNCaK9D1Z6733WOsw1lEbgzEOISWREigpMM5TVgbrHEKE6xJC4JzDWItHooUgjiVaRUghwDusAy8FkRAkiW583Xg33q/O1k/P7vvNDef4cJwMj/fDe4+xHgj3zjm8FWgJUkoiLd933LcbasbMTmgpwXuEEx/4+qPA+/AFy4uSnbzG1DUWQSRAqIgslggpiaXAOjDOsDut2JtWOAdpFC58g26MQlDYmrL06AgGScqgl+GlwFSONKZRg8sYw9aoYGc05Y1rO7x67Rq74wm7E8NoDDKC1YEg7mSAI5GKTqy5ujtib+pII+hmmklpsI5gkPU6JFoz7PZZm+twanmetYVeYwaXMYar2xNeu7zJS9euc+XKDTZ2YesqeAfKAxEoQGkwFiQwHELShXwSLnxJBFZAXUOmoT8PS13JfSsneOzUEqeW51kcdBpbOPM85y9evcL31zdY39zhtbdgtAO1g21g/1Jrgf2soQKogC5hhyqBFQndGPYKuAw4YAk4s7LDw/dt8JOPPcrTnzrdmGFirWX96hb//Luv8r2XN/nBZXilhh0gwnIfE+7PJqjoKtMCnIXSwhQwQETo2xYB8wr6GeQVxAmcWoGH14acGA44faLmM6cXGptXnud849JL/NM/XefSZmh5cpPp7Hb5w723ALpcZvy2N3hoqTur6L7GmcWjEbrXdc3G1oTN0ZSqhqqu2J3kxAoiqRiVOVPjWeikSGBjd4+8KFkedEmU5tpkgsLTS1KkkhTGowT0koz5fsqJ+R4rwx6dLG50LfTeUxtHVRvyqmZUWCQOJQSlCRtohaO0grKqMc4ipSJRktJYjPMoITCmxjqY6yckUpJbGHRi+lmKUoLSeAZd7rnB9b5hRCGEBP7Se/9EQ+M5FuwbMEIEL8r+71px7Awuax3OeYx1GOfRSiKFuOnlktah9Z0n2b6BZaylrCwIcM5hPSghibQHIfDOIfBEUbP9yuvasDet2J0UTCtHUVVMJhVSQb+TUXU0ygtqLBjHuKwY5zXeKqz3XDNTEqXZ3lPUQhIT4aVHS8g7Eg/0uxlagTGOOG5mgfHes7kz4tWrW3z/jct8+7VdtnagNJBPYZNwoT4lPEvzU3Z3wDhINNQGRoQLXJYapABfgZSGtLPH0gqszU0wZg6pNP1EsTjXjLGVFzU74wkvvfkGF1+fsH4Zdgy8BdT7T7LvfJ3IYYUwrynQB3oEj4IDlq/CyZ7j8tZl9vIxTxpPGkkGvW4T02Jjc5vvX77Ca9d3ePkVuFQEI+Pd2H7b75u3/fxDR7DCbuMqcPUa7O060uh11ha7PLh26tDG/n4URc0Lr7/ON7+3yUuX4VXC/xvC5/UD4Ac57z3Z27HcsjorWBzBq2/u8sTDu3jnWO5F3HeyGWPr9ctX+Sd/tM4/H3/wc+8Wz61pZhq+/Pgi/97PPs4Ta0cvdB9PS3bGE/YmNdZ4Nicjru/u4awhzWKqqqKqPZt7irys2BuVTCp4/cYemYZpDUUeNjNpF0wNmYI0TRhmCVd35vj06QVOLy3S7SSNXeOc8xhjmFSGaV5R1oadccm0qFDCMilrjIdBnDKpK/amOVJCbSuKyjPoZDgPAo9SmnGeI2REL+ygEdIQa4WPaspS0OkcY2PLe++EEC8IIc68W9ueTyLee+ra4mAWits3sIKBotTxMba899TWI6UIJ50Inh6tgFkY0Tp/80Ped9nW1oN3FJUD4XFeYozDeYHAUBlJlmiEgLJ2aO0bNTKL0lDVhnFRU1pBnht2yxJT1ZSVY3cs6A96RCKIU9/aHFHWnoVOB+tga1IBFYqaftanE0vKuiLWCudzvKsQQtLvxAjd3BcwzwteurzLlc0Rr1zb5bW3YMOF6/CY4DUAuOKBrdteaG79GANzRTDKKsBbOLkbwiLjScm0us58p8dOP2Nxrpl5FbXl2taYi69OeHkDrr9t+O+FJxgd+4xmt312gPEYrIdYjeh3t1nqp40ZW+vXx+yMc9avwv9X3PExHAo3AFXCySslb9wYNWZsjYqCF169xuuX4eVDfu9NwiZg/CJ0knVWl+e47+TyIR/l3fnWy6/xp0dgaL2dn+zCzz+Y8LnTw6M/GDAuanZGNaNpzlZR8ea1DbbHjnICxk3wCoQFJFgbDCulQerw99JDMQ7GFgqMgURC1i3pD8tZ+NGSxClxpIjjZjZpznvy2lKVNVd3p9zYHrFXlNSmZi+fIr1AaMX1SOIM1M5RmIrpNCeKY8ZVjbGWSGsyrfDOobUmr7o47+nEEUmkqS1UznGvC2wcxGVxCviuEOLPgZvV2Lz3Xz2yUd0j9j1YjhBaertHy3v/ge/RJM55bk9u2R9jVYeYPF7cfHzfZWuMxQtBVRlK41BSYl1NWRlq54mkIEkiytoihSBSNG5k1s5RWU9pHXXl2S0KdqYFCmCa47WnrBxCG6raMc6nCK/YrTRu5t2b5FP28oI07jKtDeOyQNWSWCmmUjGtLWZUsDRIOWCeyEdmZ1KQT6fc2Nvjyg3YdHDlLt+j4m3hEeA1oMqhm4OpHCtz11lb7h3OoA9AbSpeeusNXtyAN2djPCwuA24C2XW4sbLH9eU+D3HiEI/w3uR1yfqNnL+6fviG1j5XgY0rMJlOj+gI72R7NOYHP4TvHNH7bxE8ZN95Dc4+tHdER3kn331leqjn3nvx2i5c3ny7L/PoKOuSaztbXN7bYysveeNNx9VN2OPWeZkQQtaScJFOgAUBhQ9e1wKIZ25mQ/BknpmEF4x7NVcnE67vTpjvpc0ZW9ZRFDU7RcmNnRFX90ZMKsNoOub6To0UwXFgHXgBvRSKGvIChC+RviRKw2ZMeEhSODnsg1d4B3Eck2gdnA/uGGu2buO/PPJRHBOcC6FDOTNabvdoSTkT5x0jPKCUDCFPPGVtqa3DO0+WaIzxqFmEzFqHIxhkAsiNw9QO6w1aaeqZkTayjr71pLEKBhsS6xxKNRfLlwKMNVhjmZaO6XTKuCwoiwKlJL0owQ8UPi9xQrAzzbHeE5UFSmkqW7M7GmMry9Vkh1QnKK2orGF9b4cHk3nwHivAWH/bZ3205JXDGM9WnjOd3OnV+aisExZafwNWNvb4wiNHZR68k7o2XN4rKTlcQ2ufq8BwF87sjcinTVxOA6kWrL8R/rdHyUYBztUf/MRDYnN7xOYRe4BGwI1dqGxzn9fGRjPHeR0oqoPEWA+Hqqq4Np1yfW/C9hje2gyhXwsscKfRdcc4P8A3YAF/BfqDiuV+SlFb8tIw15BK23lP5Qw3buxybbTH+o1tcmeZjkLos64hjqCuAAG7gIzBllAWgITuAKihsjDsw6YfMc1qnDBke4o0hoVej0g3nwj1dg5SZ+sbQoj7gUe9918XQnQIWtdPHPuZblJyU3QuZpkPwotjFUKEYPw5H0KCxnrq2oSwoZaUlcE4h0DQnT3bh1dgZ8JCJwVF6fBYvIPKWiRgvKc0lm4aTo+6dkQNhhKTSKOkxFjDtMi5NhmR1yZoE4xj5CZMixwdxwyyBGMMRVVRyRKvFFUeYvpdDdd3R3i/y6DToZskGCWoKkvtLMM4QyjZmOcu0p69YsLutuHK9B0yno+MI4QjRyOozbuIpI4Ig4AKjlKZMwW29whZgA2hJew1YAMZIBLNXQw288mhn3vvhgGcbW7NdA0GHnxzexny2tLVEZ0k5q31CsEtCeR+0sKHel9CKHttD9ySxRqPp7l/orGWvDBsFgW7oymT0rK1A3keMkOlhrIEX0NegpKwvARTEbxZiYBiBGUOUoGNYcvBAgXTSDHSU4Z5ytq8mDkO7i0HaUT9N4C/STCiHwbWgP+JUOT0g177OmGTYwHjvX/6owz2qJnVtJiVFgiermBoBc+RdcFkOS6ZiaHsQfDMaC1xpaOoHdo7hIdIK7TWlMYjhUcKgXVBSB9rRWk8Wgk84JxFz7IXPYJIKYR3CKGRSjQaSowjzSCLSLSkqEsEnlgqnNIYV7E1BaNyhoDxDuEVWkm8cyivSROFVA58cFXXBWzXU2y34qGTJ9Bag/MICUqIxpaXTqTJreX61jsF1YeJtVA0aGyJoFLlKO2SFFAKRIM71K3xtJFzY9/T3BRVYRoxthJANrhB7SQcTNR/CJgGl3/vJd00o19W6KRCE7Jfcz7ahq0mGGt7Y8jLAh1B3GAEoyxrxmWFrSy1sUzz4MWajqGsQ4kOqcHWkKWQJiFbWxlI45DJHOtgYDgPO3swPwdSSbRUOCFC5rpSDZqQ781Bwoj/MfATwL8B8N6/LIRYuYtjfMl7f+PDDK5ppBR3eLSkhFnsDYQ4dpmJ3kOkBWXlyWuH9YIk1ljrKK3HOEdXeoTwOAdOgBASrTzWeSLh0ZHEI6lRRFEIMapZzLQ0njhySKEbPVmVkqRRRL+T0EsSatdne1qiaoNzoETIzgPBznQK1jJINLkXZFGElAkd7xgbS1TmxF1FFCm8h6I2FFXNuCiZq1JUJhoLD0ulWenESB8WuSM5BmAkuAYv3kp5dHJ0uiYIF5ZYQqyaa8t6fXfSiAu/AKZlc5qtOJGNfJ8tEDeYbp+kNGZsxQ3GdobdBOUN26MR+V7IFj2saWYE7xFC0Itj4oYyz733lM6DAyMF07rGu1AqpvIwKWDZwcICmCTotvBgK4gcVA5cHh4bLoWSLFUNtYXaOyIdIaWidh7vHMdBbn0QM7b03t8MvAshNBwLQ/HQCR4tccvDRdAP3e7JEkIgRPB63Ws8t2pEKQFZHOER1A68c+RVySgvqSob5gA4Z0KZB2dRUtBJE2IVit71Eg0SKuNQUpBogXcho7HJs1UIQRwrpHNs5RMmk5JYC5IowrvwmQgBuTE469iZwrWRQUmBwOGlI0li5mKN0BGVM9SmoihKtsdT9qZjqqrCeIv34fNtAus8QsXEnVBn6SiIgESFHWBjeEUvOtpFoQfkNUSqufOwMEGHdtRkhE1AU2QqogmTNQF0gxvS5sxV6CXNfcEWeimjqmBz0yPU4Wp4PNBNoZ/ExJFAyWasSOc8WgicAGnszRp8UgWjJAamFRRFMKJ7aVjTqiJ45Dq94DxwBibjWXKAD0bZaBy+u70koSoshbEcg0DUgTxb3xBC/B0gE0L8IvAfAb99wPf3wB8KITzwD7z3//DtTxBC/E1CmJIzZ84c8G2PDiHu1GYZ697hwToumYn7RiFipscSoW5JWVYgFJ5QCqI0FoQn1ppIKYwDhcMTamlJGU4Ei6CjFVaFysMoidbNh0y995RlzW5t6UcJVaLAWWoXasRICVrD3riiciG92erwRZuUNboSTJUBa0iEYGxha+TpprCiJZOyZlpaTG1CLTHR1G7OUdQFtghiz6NAASuLkDboASrqkr0iGHpHRQ3gwYvmvneJlOy8W4GwQ6YHJLq5SutCHWyX/VHxgDtirczGTs5vvbDBcxfXeam5BEF8gxo7pRTeeerZnrfHrZpfH4X9Va/fBy80e3ndmFHigTTWYAw7ZUnlPFEH5BTm5oImy+SzjTXBa55EIQGnLiDqhnN4PIGkE4oix3G4LsxUIkyKAiEc2JBZf685yFXmN4D/APg28B8C/wz4Xw74/l/03m/Mwo5fE0L8lff+j29/wswA+4cATz/99L23YN7G7Tquffa9XvcaKQW1cThrbwqiJYJYK3LjSLUgiSS1teRlTS+N0ErhjcN5iZgZjJHWeGfwCCojUcIjETczMJvO5LA2FCqNpGJhMCQ322zlBXVVYCwMe4I01lRVTaagCnYm48qgXfDsFLNWKvODGAFBnyVhWhgiHeOMo5oVgm0KZx27ecHW3uEslu9GBsx1IjrZ4bYLeT/GRUFxxAL5grC7LZtwNc3IkgR7pEq0wARYHGRHfpx96sod6We1zxSoi8PPRtzNa37v25e5cGmdf/Pa1j0JETnf3BVgUhgEIZt8Whxexq8heJA8MMmnFGWJs83ID/a7sqAV/ViFrhijmk4/DKw24DqQxSAi8ONbL0x7oUtIKUKdOuWh0w2hR23DOl9XNbl3LElw4niUEjhINqID/ufZ7a7w3m/M7q8JIZ4jaL/++P1fdbx4u47Le4/3HKvMRCkEUkicc0glkC70ldJKkcSa2EmK2mK9x9YOIQVKhDpiDoikIK+CVkqpfcNyVqNrNucmdwbGOior6GcZeWnJ4piec4ieQ4gRUitMVZNEIZw4LQlieBF2NSqJSKuaSkJRVtQ5YEMYyjpLohTGwzivsa45IXlhQ//Go4q3JQTPVm1co81vayPBHW0YpwM4CaVtztpSUh6pDm0fAyz1mynUCpDXppHwqAHKQzINSmP5F391nQsX1/mjv7oaCjPPWOjG/NqTp1h/9Yd8/e1F6I4K2ZyFtzuesDku8D50mzjM5IYFCVZBZR3Whoz0Jti/rg6SmLWFpVBFvtxhMtMW13XQxVkL0gXp9DQPUQ1v4XoVNtXGBwG9nUKWgJfgHGRJTF9JVJySaomz996Pc5BsxF8F/i5w/+z5B2pELYToAtJ7P5r9/EvAf/XRh9wst2cm7nu0lDoe2Yih/pdAR5oMw7iwmMpQ2ppenBDFEVJI0JKuENS1RUiIZWjg7ADvPEVlEBKk9EGkiAhZlyIYXNZ6dEMtbfbnFatQSb42jlTH2Nizu7tLJ8tIhGDHOjpacmNsqHOwEQgT7rspIKCrJTuVAxWyV7SE0oWWRIUxOOFw7p2ey6Mizw1SSLziXdvXfFRqgts9r0Mpj6aIIk9ZcqQ6oJiQeaQbCYAFrPcNBBFhAEjdXEss5w1NnB4DwNYf3uh3zvPnr2/x/KV1fvcvL7NX3DJ900jyy589yfmza/z0o0tESvKf/f0fHsKoD4ZsyAMEMK4KPJJOciuD8DCICPu+ugih5UgKqoaymINcR5ImEVpL0jghIkhC9vZCMkySwWQUQoI6Cp0JShkah5sa5CCs6aYiCO1taG2WSDDO4xDI2dWMhnS578dBvuH/PfDrwLf93QmVTgDPzS5iGvjH3vvfv/sh3nveruM6Lux/GMY4KuNJk5g0iWGU44XAWYuXoIVCRhrrHJGSWM+sa3owsMo6dJGflpZICqJIhE7sDrqxvq1lUTNIKUi0Yq90mLqmNBXXd/eoJQyiGDPTqTEzlpJucK8LG1KE89qgpcBIQaohUYJx4ZEKYm/J65JeJyaLNLrBOlsOj3OWo3KmJYTdXW1gWjdXJLOrI3RytHX4HdCNoNNtrumGloqUowv57iMJXtamiFRME0HLHEDe/cn+0pURFy6t8/yldTZ2bvlxpICffnSZ82dX+aXPngwJPbchGkwKsQ1qB71RzHV7vHW9uOutRijL/e7UhM2nF2ETU7pQU7Ip1M0C4rOErCTG5RXYMK4OEKehjpZ14CNQNhhXdQm7NaQpIIMxJgn3KIiUxjmHjhSlsTclM/eSg6yPbwLfuUtDC+/9D4CnPtSoWg6EYF/Az83SFADdVN/MKFRCEs08cbEKaXzOOJyzQSDvwwkvhMR5S20tDolWMw+YFI16f4DQTFtK5lPJNBJM9grSLKZjI0oEdT5h2OuyO5qSxWFxsOXMU+dD1p8ToKxFyrAwilkhPCRU3jGXJsQ6Cl/2RmYFUjgmpj4yb4kluNG1grxs7uKdZhlrc3DpCI+hCDvc+aw54b+SspFGTopQhb8pOklML4GjjiVWQHzAJePybs5vXdrgwqV1Xrw8uuOxJ9eGnD+3xq8+dYqV/nurzZosx+Bscx7WXi+iE0mUhvkEsvLgpR8+aG3TgBbgNUjcAV5xeEghkEphrCeSEVmaEuuKNAPjgj5zP7puCNqsqgobymRWWV6n0O2GivIKUC6UQ1JAHEUkSmO8PxYJbQdZS/428M+EEN/gtq+n9/7vHdmoWg6ElAJnACGItcRYhxeCNI6ItMd7h9YSLeXNshbGerJYUBlgVjG4kyhq69FS4mdhUykUSgVDK46aLWqqlAQRylIszQ3Iq2BAldbhRnu4NCWVCjWn6dc146Im0gbjPJGUeOlQ0iO8II0itIwo4worFYNEs9LtoWONUgrrXGPaSYGkE6UMOhMWp6Fp72HRB+YJO8BuEjfqidRS0O8PWR3sUu7BYQdzHgB6OhQ07WTNuS+yNGKeu+9fedfHEaAb9Jz3OgmdAaFj+BGyGgP6vY3jvaLm9799hecurvOvX9u8Q+h+ZqHD+bOrPHNujYcP2Oczbi4nBHlQK/IQWOhkqCRmqS8oTniW34A3Zo/t25d29nM2+/l2Y2zILe+snt1iQpXy5QXo9qETpURRhGgwTC+UopdoIh26hvSzDmtLmp1sinWWSV7T7yTEScnerI9jloGcCzqt3d2QuYiCNAsaLiFgvtenN+jRi2OE1qRKcQwkWwcytv5rwmeVEj6jlmOCEIJICao6eLeUkqHWlBBIa8ErpAx/2+9tqCSzSvKeLJ7VfpIS7y1aKeraom8+V4ZekbOm3E3OK1aSSiuyOGZxkLA7NnhXMeh3Uc4xNZaBjClTgWeHQZpRlTVGejpRjJSOLIkpq5AE0Ov20Dh0nLAy7JNECVmkMDPdWxMoHXFiMGTt9IT6DegWtxbNj0KHELM/1Yc4g7lej37WXCmBSCvme30eOrOLuAzVZmggfRisAff14cQJeHhlBd9Q03CAXtbn9NxVXtw52uOcOAG9BsOjw16PM4vwqevw/SM6xmPAA/dDN7rzPCyN5V++dJ3nL63z9RevUd1WfHe+E/FrT63yzNk1Pn9m7q43DN3BEYkh38YSsNx5X8nyodLrdljr98kX5sn9Do9NHfNbMHUhk7VPaE7fJ1zQC4KBlRC8/Sc07Jhw8e7qILLPMuj1oduB++YzTs7N0c1itG7OPSgBISWDTpdRAQMknlCyqKgNERPm+z2cTdB6jFYaJTSlsVR1RZaGzilJrEmVJopiirpiedhjsdOnkyrmOgk6CqUz7jUHWbkWvPe/dOQjaflQKCXRs9CZnhVbtdahpSSJ1ay3Y+h4tV/GYX8Rq22oLh8sNRkMsEQjpQwuXnlngdcmiSOFqixJJFjo93DkqALSyuKERxtHJ9F4F+rfCzSu50hjTSQiKlcTSY3LLMZaIh1h8CxmGcN+hzSWKK2JlGzMC9RNIoa9Lg+fnMP5HTrXYbAT+pNVzMJJs/uEsGhqwuMxM88VYdfjCJ6x0wSh61wMgwV4ZFlxYm6OhX5zF+84illdmmMvz6ndJkkEwyuzhr2zsb89J01yaxfO7Hmd2XMns7/fD9w3D6dOwCMn5lhbWkQ02ONsea7HA6dhbefomlF/Cnjs/oTFwcG8N4fBsNPhkfuWuLJ3g8nG4c/tNPDQCqwsZpxa6uOc55s/3ObCTOi+e5s+LY0kv/iZk5w/u8rPfGqZ6CO0i7lvaYnPcJXvHcIc3o+fWITTJ5eP+Ci3iLRm9cSA2nlimbGQbfPWzpTRNuQFWAHZ9qxAaTdUUY91qKgeSZhbhrkpRAlYQ0ge6kDaEQxizZmVFeY7GUkchdpXDaG1RJSCfqZJI0VpSqSAVINEkSYDpNL0eylz3RSLxzsFOMZFiXcOnSgSIUniDsJbap8wl3XpdRJ6WUSkFQKJ/jg0oga+LoT4Je/9Hx75aFruGiEEkZZIO2swLUNIUd1mRLxb+E8pifMO5zzG+ZvZh0qGUJRUHu9Dra17UepCa0U31RhriSNYYGmB7AAAIABJREFU7idMYsG4NEjngiEpFc7DsBdjncd4B04Sx5K6doS6gwKHQPlgUHZjhY41/VQRyVCTrCmGnZSluS5FWWKMR6ld5ubhMQ9ilt7sCsg99BMYl8FVfsbA0hzoWVEcFQcRfD0FFMQ9WOzDyYV57h/2OL0yYNBtrm5TohUn5rpU1QpISb+zS9apOHEDjAnhHQtQwe4slUrPNHSOkHmUaugMQumOIofRLvQX4f5VwenlRVaHc/R7KZ0GhTkn5/p89oFV3tzYQG6FcOJhKuE+Azz1MHzmzBoLTXq2uimPrJ5gazKhynPU9uF4WAEeAj59Eh5dU/R6C/xff7nF7/+f32d951ZgSwr44iNLnD+7xi8/8U6h+4flM/et8GOPXeXqS4cbor+ds8DnP9vl9OLwiI7wTrSSzGUZkwULIvSO7cR7TIYFwnmMsFDUrI9DCQTdDWE2oaETB42TXBRoKRBeYI1DJhGZjJgfdJnPMnq9mMVeQho3F6aXUtLLNFWdsjio0crTURGTWOO8p6MVURTRy2KEgMmkIDeOJI6JlWd7UqGlp9dJEE5g8GRakWUJkQ4GXKwEnUQRNbjOvxcH7Y34t4UQJWGtOVDph5bmEEKgtbqrAMu+keacQHmPd+Jm6opW4V7MQo73ovG2EII0iVBSMC0MNtXM91I8jrJyhGxeiVYCaz25MUjv8VIinMN4gTMGI8AbT+UN1kjSRNKJdfCA6dAzsSmyLOH+5SGxlgy7GSfnUramBcM44cTCgG6aYj2MxxNu5FPKwqKkIdUaZEqsLVKm4GoK71B1zVZdM4hiBr0e/TSl3+twZrFH3OCimSSafqfD6oqgm0Ys9rosdXcxp2t8bUN1aCWQUlIVFbt1jTWhhk6sYa6XsDw3IJGa7aqinBZ4V6OimPlen8XBgJX5Dgu9Lt0GBfKDfpenHz2D8IbB967x+noQ6M7kI6wkMDeA3TGMZraEJHjp7OxeAnvMvMqz3/uEkO9DD8AXHj3D42dO0e02F/btZAn3n1hECpjvXGb5jS0u3wBdQS2ACgobvIylDI1/Kx8KB7vZPCaz95pPoZOGxIx+D/pDuFb0uPCDmh9u3+kz+9zagGfOrvHVp1ZZGRz+fM+cXOGXzj1KJF7mW38VegjuS+0joMus0Cp3LwO/D/jMApx9LOHfevxRFob9wxv4B6CUpJMlrHpPEkmyOGJxkIYwnFbEMoTjynzK+t4YUzmssMQInNDEkWI+TbEqZNdapbBVhZGK5W5Kt5sx10mY62WNhhEBtNYsDjvEWrKVJeR1jbUuZMNISaoVnSxCS8m4qDHGIGXIIF8cWmItSXTIUJdAFEfgfNAuR4okUiSRvimjuZeI46DS3+fpp5/23/zmN+/1MFqOGd77O0Kh+70pjXU3a41JMctE9B5nQ+FWZq+z3t9MaQ6JAop4ZmjJBsNSEEpulGVNXhmMcwjvQthWKSQePNTOUVUGlCBCEs+8OWVtQugXTxJrEh3hvaWoQiujWAt6aUzSsEAewFpLXlRMK0NdmVCJ0Aqc9Hjrb1Zwdj7MrbIOpRTdWJNECougqixilgWrlcThgwdQKzqRppPF6AbrUe3PazItuLa9x+XtXSZlTSQgjSVSxSilQ6aYhFFeMypKrA0XAS0lxnnyvGRU5xR5jZGOXpRyYmGOtYV5VuYHdDspqsGGzfvzyvOKUVmyO5qwM5mQlxYpoJNqYqkpnSEvLHh3U8Bf1AaPJFaSTqpQImbqHH/++h5//INdLr01vsOQOT2f8ey5NZ45u8YjK0cfKi2KgitbO7x+ZZtre3tESrLQ7dDvRNRWsDUaM8pzlNQob7CAtZLCTqhyQ14ZJiVkEcSxIIs1Ok6Jo4TTi3M8dHKJxbkBUdRk89GwBlrrqI2lMoZ6JqeQiLBp1AopBMZapkUV+sYah44EWiiUUrdKTQkJPqyREkUUhd64Wqt7Vj/Se48xlqq2NzPstQpylv1Me8H+eh6WEz3LsnezUkbOOUIb3xCl0TqsI+qIpSJCiG9575/+wOd9kLElhPiZd/v729vuHAatsdXS0tJy/KmM4xvfv86FS+t8/XtXKW8Tus9lEV958hS//vk1Pn9m/lgUgG5pOSoOamwdZJv4t277OSW03PkW8PMfcmwtLS0tLR8zvPd8ayZ0/50XLrNzm9A90ZJf+MwJnj27xs98apn4GAiSW1qOEwfpjfhrt/8uhLgP+M0jG1FLS0tLy7HhlWtjnr+0zoWL67y5fUvoLgR88eElnjm7yl9/4iT9tNnQWkvLx4kPI4B4C3jisAfS0tLS0nI8uLZX8FsvbPD8pQ2+vb57x2OfXR1w/uwav/bUKieHzQn7W1o+zhykEfXf51byhiRkv75wlINqaWlpaWmWcWn4g+9c4cKldf7slRvcXgdybS7jmbOrPHtujUdPNJeJ19LySeEgnq3bFesG+D+89392RONpaWlpaWmI2jr+5OXrPHdxg6997wpFfUvoPpwJ3Z89t8aPnZlvrNNCS8snkYNotv5REwNpaWlpaTl6vPdcfHOHCxeD0H1requ+f6wlv/j4CZ45u8rPPbbSCt1bWg6Jg4QRvwj8F4TuGZpbRU0fOtqhtbS0tLQcFj+4PubCpQ2ev7jOD7emN/8uBPzUQ4ucP7fGX3/iJINW6N7ScugcJIz4vwL/CaHcw9F3+WxpaWlpORSuj0p++4UNnr+0zgtv3Sl0f/xUn/Nn1/jq2VVODZtr79TS8qPIQYytXe/97x35SFpaWlpaPjKT0vCH37vCcxc3+LNXbmBvU7qvzqU8c3aN82fXeOxkK3RvaWmKgxhb/0II8d8C/xQo9//ovf+LIxtVS0tLS8uBMdbxJ6/c4MLFdf7wu1fJ61tBiEGq+cqTpzh/do0ff2ChFbq3tNwDDmJsfWF2f3s5ek9bQb6lpaXlnuG959KbOzx/aYPffmGDzcktoXukBL/w+AmeObvGlz69TNJwg+GWlpY7OUg24peaGEhLS0tLywfz2o0JFy6u8/yldV7fnN7x2E8+tMD5s2t8+XOnGGat0L2l5bjwnsaWEOLfAf6x9969x+MPA6e89396VINraWlpaYEb45LfeWGD5y5t8MKbO3c89tjJPs+eW+OrT62yOtcK3VtajiPv59laBC4KIb5FyES8TmhE/Qjws8AN4DeOfIQtLS0tP4JMK8PXvneV5y6u8ycv3yl0PzVM+erZVc6fXePxU4N7OMqWlpaD8J7Glvf+fxBC/I8EbdYXgSeBHHgR+He99280M8SWlpaWHw2MdfzpKzd4/tIGf/DdK0yrW0L3fqr5yudO8czZNb7wYCt0b2n5OPG+mi3vvQW+Nrt9IvHe45zHE6q1SikQ4uO3iHnvsdbd3P1KQZiHEOBnO2IhZhVpHVVlqZwD54kjRRxplJIIIY7F/8QYQ1EaaufB2dnYJMbUlJXFiHDyxloilcY7j3WGojIUxqKlYpDFdNII7yW1c9R1hbEeoRSpknSyGK0/TC/2D49zjjwv2c0LtncnbI/3yGtPLBVZ4gFN5QRJJJnrJKRxjEdS2xprw2dcmQrrJVJAJEEqjZSKbhoz7KR0sgSlmhVEe+8py4rt0YRrO2PGRYEUkkEnppsmKKmp7eyz855ISjpJhFIC5yXOOTyWunIUzoAVxLEg0RFpktJJNJ20+c/Le48xlqq2lFVNXpcUpcUDWaxJI4X1nqJy1KZGSkEcRURC4IWnqj3GWZwzGOOpvUN4yNKYfprSyxLiWPOdjRHPXVx/V6H7z396hWfPrfFzj62QRofzudZ1zbSoKY3D1CVFVTEqDFVtiBQkcYSWGq0lkVKAxyGwxlKakv+fvTeLtexK7/t+a9h7n/GeO9RcHLpI9sBmq8nqbsWaE9gtyUO3xEYAI3ESx0gAw0BiKwmCwI4fFMcPQQwjsZM8GRmRIH6QEbFltSXBsdxtDWlJ3SCbpNgciizWdOfpjHtaQx7WPvfeKlYVb5P37Eu2zh84uPecvc9ea5299lr/9X3/71tFIfDSobxDqgitNe1Y02k1SOIYreTBeFInpmOXdY6iKCmMC7+/tYBEKkEkJUpKnPcUpaFwFm8hiiSxViipkFIi8HgP1nu88+hIEStFFIXjp9Eu5z3OOqxzGONweASglEQrdfC7AwdjON5jrSUrDHlpEQIasaYRR2itPhLz3cPmnHvnNiXFqfStD4t6R66PGLz3YfKtiMn0vVa87438KBCSaT3CBBweQCkEznvSrAThER4KZ5FIGrHCO8eo8GjpkVJhjCUdpsQK4kjTCI3He48XEgnEWtFsRLUNMMYYNvcnDEYjdsYFw3GKEg6NY32QkTtDR8KgLHFecKbTppVEDCcTdguDNqASSVdLRNTiXDsmL0vWhmPwivMLbc4udTnTW+DcYqu2Cdw5x8b2Ht9+/Rq/+71tvnMHbhIewjZQVH+7hD6lgIXq/63qPUBSfSerPnfAReAz5+HHPtflJz73NJfPLddGuLz3jCcZb9ze4HvXb3Jja8juPqQD8BIiDa0oZETOChAOkgYoDQsNaCYwKmAwCo2VhGNCwPmzgmcuPsojF5ZZWexyrlff/fLe0x+MePPOOt+/tcH1jT77u6AkoMClYDS0JKgIjIPMgPbgBHgXzs3K8FvkHqSHuAmXzsHZcz1W05j/78aIm3vpXWX/qSvLPH/1Mn/+cxfptU5W6F6WJW+vbvPmrRu8+u4O19+BrQm0AKshNzAt0RJCzxdi6HbD/TQlFAZGI5gQ+mIvhkjB4jI880SLz115hKcunWOpt1DbuOi9Jy8Mo8mEOztD1nb3GU9SjAeEoxE16cQRJZJYGHILaV6QGUPkHRPriJWk24yJhCI1JcY5BJJWHLPUW+D8YoulTpuFTqO28XDariwvGKQFo0lGbh1aQmkcWWnAQ6IFkZZIGV4NrYkiSVYYBplF4gFPmltK6+g0JWe6HRa7LZpxRBTpU5vDjA11c85jrMM5T6QEQgqKaoEmK8OBIxgTtBQIKREepBIIwvlSiI+k0eRPFNm6lyB57xHi8KaEv+EcpR58oz4MSTtJTMt1zgfS5aD0lvEkY2N/RH88wdhAtGItcITBIYkUSaRBRnjvmKQ5hTFEkUYLD1rRTRr0Ok2akWLqyWg141raNxhNuL09YLM/YGc8YTgYcX19TDaGhV6Y0AYDSFNoN8DaAWkKgxKW2rC8CMMhDMfQa4BKwBpIWrC8BDebEReWe3z6Qk4rPs/iQmfmbQIYDEf81nf/mBe+NeSlI5+XwFTyPCEQqB8UfeD1DfjtjSF/ZfMP+ct/7ic4s7z0Yat8LBhjubW5w++9+hbfe7NgrQ/vEMggBPJ03yib6hgEImnvc97yDc+zCzf5M1/s8/QnHqcVydru13g84Ttvr/K967d4+2bOH2/AjZO48LB6vX13RvdPne/w/NXL/OJzl7k8Q6H72tY2v/3KW7z05pA3N+HdowfNA75UADsPuejUGHcHXr4z4Sc33+SLTxf8zGcfpbdQj6asLA27wwnX13d54/Zt3rw94PY6jEdhvI8iuLgSxoO8IIzXGiYDyBwoH8ixNYAME3q7A80GXFhpcS4L1s3MBItju11PMEJZGgaTgv54wvYgY2e/z1p/gClySifpNiOSRhPngiegqSSxbhBHnrR0CG/xXlDkGX3jaEiJFwItPSu9Ja6cWWCp12Gp3aCRRKfgxbDkhSEtLaWxSAHWgZSgKkIrpcIKT1k6nDekhQHvUUKiFXipaUcCKRRCCYQXNBN1agTyfvgTQ7buR5BK64n03Tdieuxh1ykKg3GBqB2aNHlfknbSCCTLkRUlw0lBYQ37wwlruwMGRck4zXDGY3H0JyOaUUI7ifFS0o41rTjGWkApvPW0Y89ulqLxrCz0cN6TNxostiIKA42a2rczTNna2+faxjbr+2O2tuHNbVgHlrbCBL1LGN+baZigPWGeWBqDH4djI0CPwYzDdXsDuLgOl8+VjMfbeO85t9ipbfJ+d2Obb33nbqJ10hgA//sr8Pkn3+HLP/bFGZZ0iLwwvHVngz++XvBuP0zeR/f1ehDROnrsQefsAv9yAJPf7ZNEa6x0m7Xdr83BiDdu3OC1t0te2n041/igkMAzy/Af/ZlP8/NfeLKWieH1Wxu88faQVzZhbQbXfwsQb0Cr8S6PnWnxbE1kK81LNnb7fP/Obf7wtQHrm3Cb8EwYgBLidbhMsNwZQj8VhEXOPoecUREmx8V9OA+M8gmlKWlGkna7ySDNayNbWW5Ii5y1nSEbewNu7g8YjcfkpSNWsD6GpuyDdCAksVA0dMzYZAgv6TQ0k7zk9o5BC+i2Ca7uKKK0A6TweKHQEiKt0DXmZDPGsD/OyQpDYSzOQeEckRJ4C9aXSBEsdcZYvIBxVuARSCReQjuKaMSwlTmiSNNNgiSmP3G0E08j0bW7fe+H42xE/Z/d5+M+8F3v/SznjROFc4dEC8JfKcBad1fn8t4fWL2O+ryhmtCNozzyndJ6nHdEOhhp64R1jjS3TDJLbh17w5zV3TGbe0Ny73AOMlMwHI7pT2Cxa8mMoT/MscBiArrVJlFgvWApaWG8RwvF9miCc56FrkOJBovtZm3tG0xSrm1t8+76mNVtuDU6nBQ27zk3vef9vRPi0YV6v3qZTcjHoNUOW5dXeJLzJ1j7B+Pa2hYvjmdfzgD4nVfW+fKPzb4sgKwsefP2Om9tB4vWLPAHJVx+bYtPPXoeODOjUu7G2uaAN1ZL3poR0YJAMtUu2Lxf2wr8xvou76zNhmhN8RZwZRVWt4Y8+9QMCzqCcZZzY3OfazcHXNsMY8XonnMK4PoxrmWr10b1Km5BmZW0mwPOLfbIyvq2CS6cZXeQsTPM2MsK8qJgWDj2dsD6QAylDn6LRtehrcOJElOCTizrOyWjCYyGsLQS3PVxo6TbKoOvG087bhArwUKrWRvZstayP85J84JJ4ShLixWCPC+REqSUpFmO9dCMI3LrsIVlkhe0GxFCKpSQDF0ZDB1aooRglBuacYwSPljKpCSJT9+teBzL1peq1z+t3v8F4I+AvyaE+BXv/d+bVeVOEp73uviUkpTGHbgTvQ+CSFn5wr33BwRLymDFctU53vvKN36om9KqXvZsjcMYg8cxnKSs7fYZjifsT8akVf33BiXjEaDAWs8gzsN7B6MI4vEYISCOoK9GtBsx1ivOuC5SSOJIs2En9BoRdXXVosxZ3RizsQfpKOQYOUlsAukYlgcwHN5L12aHOxvb7L//aSeCa8eZUU4Iw9GYWzuznbwBvnMbvja+d/qcHbZHu6zdDrq6WWIT2NgfzLiUQ4xNytszLsMD72zB/qS++zVKUzb299kZhfJPsuS3gHQLegt9rpxd5PGz9e0r6a1jmGY4HIUp2RrmbGzA+iAQwjYhUKaZwCSHMg+aR0zQEQ7ycF4EqAlMhmGOy3qAzGgmEZuDEbGGSysLkMw+Ga73njQ3lMaDkKRFSV5a8jyntBbvJEp5jDU4B3mW4z2MioLClDjfpp0kyFiSpQWj4ZhOp0VWWlqRopXEIATGC+JjSIPqwHHI1grwBe/9CEAI8cvAPwF+hpB/62NBto5qtI5Gdkg8+KNCdyhLS2E9qgqtFlLgfOj0Qkq08FjrESJcz3mP89Qeiu0F5NYzygyDccFwkrIxHLKxm5FbsBayHPr7EMdQFNDrhXamGZg9aHahyGFpGUYeBnFBowGNROFTgfaOXifB+G5t7RMoZAR2EiwK5Qlff0BY4WYF5O5BIpWTR1G8/zknhdHDfHcnjL0sRdr3WhlPGqvA/nDyvuedFNLUMKrBnLtLsLzXhY6STB7q3D0ZbAKRqK8jFmWIQHB5eMZPEpbgknx8DXafGNNpxidcwoOhoxAxWJqCrcGA7U0YDgKZHBPa2nCQpNBJw3ipqmOCQLLy6rzhfmiLtFBuQ6sJRcfRH49Zammsq+d+hYhRj5KwMyzwPhgONocjpIdO0qQ0FuvBmJzUOTpRxH6aoZwDMgpTIMYSY0uMt5TWEOmYtJkgZdCyRVKAp3av0/1wHLL1GIeubAj38nHvfSqEyB/wnY8cpBSUxuGcxTgQBMuUUhKPQFdEwliPcf4gfLY0wUUoRLAkqYpgKRU6snMO4X11Tr1kS3hwpmCQpgzHGZMyZ2d/TH8Io3EVsZZDagKx8IAaQl6Ct7DQAS2BCPb3Q6SUimGxC/tJSieKyJzmQtwi6BF9LW1sNhIuLMS85QrWZ1RGBqQTcKY+slWnbKDOPOKToUHU0DYDDPP6hhyLq4GShMlwsV3f5I3W3D2kzwYKiOI6SYlEOomYkRLZArsTyNIRcY0pSCKp6DY0hcmYjB07/UOiNSYQKYC4em8IY7+rXiVhvJvq1MbVew9cLMDjyY3DWc9D5MoniuBpgrw0TPKCLLekecEoG4OXxFoTRwphHHtlhnCSUmuUdyFlUZ4yyoKExzrHYiNmc5Sx2PI0kohRmiGEoJVEIKjNK/MwHKfH/N/At4UQX6/efxX4x0KINvDa+31ZCKGA7wB3vPdf+cA1PSG4StguONRueR+iHKb9zFe5TBAC7xzGeLRWB+GkxjqUDOGluOCOVDW7ECGEu+YOsnHBXj7hznafSRkibLSCNIckBudDdIcj/J3KDYyHlg4dXyooM4hEiNQpC4sTilhJBEE8aYwjjmffzl4rJo401hY0COLVWSCdBN1AXajzgT+zWF9ZVuRszUrUdA+yrD63r5L+RF1RD8JlYKVTXxb4JNb0KOi//6kfrhxCmo+6EAtPZkqaOhCPWUABgzSjrHHckKLSLxWW0oZxJCPQ5ZjgRtSEzzUh6CKq/mrC+JlXdVfV9zywCDgDe5MJSSTx0tfovQDhPcO0oCgMRenIjCVWERKJVoFwZa4kzx1xpCgNOARZWVB6j9IR7TgitY7SKxZbCYlSZKXDuYKlToJ1Frz6SCQAPs5G1H9XCPEbhCzyAvhr3vvvVIf/nWOU8UuErPOnuqeEc6EjqSNJ3KZaK+eDhWqaALQwDiEFWimkkuRV7oMkqm6aF4QsESEXyKklWPOg8KQ2ZzgcUxiLr3IY2SKIJyXQaoMrwMhAqhox2LIaOPqwsAKdGIY6DI5RArHSICRFacicJdKqllU+QCPSJJHGEQbsWZEtCZgaKVBanweMTqu+soQT3H8H1ZNHYevzxUZa1xKufaENUZLUUFKAljF6Zk/VIQaEvFx1wXpot1tEjXQmZKtFGEvxgnFW0Ou2Z1DKe1FaS+lAeEdRuQlNVZ8mwVKVEsbKBmFcny5JRhySqzHTMa+6LiBjKAvIyxJhfW26YylFsLoZGxIBxzHWO7Ts4KxByypHXZETaY9UAoUPec+8ZVJazrZa6ChiIZJEOqLVaCEI83QzBoGuUkicPtGCwzQ374cXgV8B/h9gUwjx2HG+JIR4hCCo/58/WPVODlOB/FS7RfXeWIexropOFNip5ctzEIUY6/BDySphWiPRJLEOHVOIgzxXdcNVWjNrBEnSZLm7QLMjMTYIIxFBqyWAYRksV3hQKli1ijIkKLRlcDtKER6+bhuaSYxSmjiOaGodyGpN7cqtoxm3Ob948nqtoxg5EDWuUCf1eSyRdXZHqahDe9oBjJl9Qd57Xr3T5+uvTbgz89LCGq9O7a6WjjqoXQGI+h4v0tKxFDVox7OxbIUdK6DRiMhrjEac5CXWOLSKoJKvFIRF6B5hjGwTiFYEdKJAxEbV8VF17piQ3m06pvaB9TUoDTS0REZhfqsD07m4lWh0lBBHkl63Q0NFeCkRUobANCFoxQ2aOibSIY+WljHL3QUSHVEYR25LjClIiwJ8yJAf64g4FjS0QilZqybyQThO6oe/DvwyIQJ2mpbEE/ZKfD/8A+C/ICTFftD1/yrwVwEee+xYHO4DYUqypBQH2WohpE+QUh24AT0iRC34IJC3zqOVOshMK6p0EaX1SEGl+eKUkpoSzK2RYDwZsJumFGNwOeQTQMIkCzetrATM2ShYtlQJqg2LC0FIn5ew3As6sCKDXJe4IoNGl0akcAi0rodujSYFYBEzdAdAGLBMjX6OmrSnAIxnyVLvgZJgapBSLXH81eEHwa3dCV9/6Q4vvLTKtc36ougGo3rHDbyqQbFVTepRfaYtawx7Zc7+JFh8polyTwqa0NejqMHDs8edLPLSMEpzxlVEXkqwTkmCe3A6iWeE8bIsA7lqEAhVVNX26G9RVq8cECU0Gy2USGoTyENwjTaiiHZSMMkFHmi1YtykpBFrlITcRCipiJRmVGQ0Io2QhqaOaDdaJK5gdzimKHKMczQWOkRSoJUgUvoguOA0jCH34jhW8l8CPu29/4FUGUKIrwCb3vvvCiH+jQed573/R8A/AvjSl740s19kSrKEAK3EgftQiUCupoNdpIK1CxH2YBIVwZpug1OWHin8QZK0QLLEqSQ1FSIkvLNW0Gg06RrPeJSRp4fHkVXESRmsHR6IFZQSmiJssdFMgotL6/BQagVay7AtAg7jHM1I1Lddj7eMspxJykxTJZxLIFZJbcL/VmPmRRygrNGKFinFoIaxrAUsLpys62ZvXPCNV9b4+kt3+KN39+46dr6jUCPL6omW+F7kgKnxhpW2rIUqSCCZlVr9fhCW7f0+W1vBgnPSticLtBS0k0ZtVn4IJHIwSRmVJcYEEgWHmq2IYLWaRtTDIQk7Rzh/lff+HlPSNs5Be89CK6IoLfU4R8N2O0pJYh3hvKE0jnGRQyXhUQq6zQbGh7RFHSBRltQIGjqiGWkK62gkmoZssNhp0Wk1qr1IFcJbBNFB7szTxnGehFvwgbSUPwn8ghDizxPu94IQ4v/y3v+7H+BaHxpCCLTirg0tIyXwUmHcYaSdlAJvQAuwlXtQyrCBadB9BQuFPpicpzk8ZO3sWQCFC5vfLiQN+pMx3oZcKyYL+oJGG6yAbgdKD64M+7Z1YoiTcHypGTNuFZgyXDSKIKp+i06coGuOtNSJPXuCAAAgAElEQVQI9sdj8mJ268cloJFAHKnaSPLFlZNeaz8YcY2ji5CilsFMAyvdDy/9zErLv/j+Jr/64h2+9eYmpT18bs92E37h2Ut87eplVm+/w3//q6szJ1seGBX1RVmOy7yWXigBoeqzlJSlpyxLivyDTVjvh4P9SxOF0rPPRTWFF+DwB9te6epVEiZWx6GGCwKJOuoRENw/OEcSRiNTQCEgVhG2xjlMKUkSaQQeaxz9yYS8LGmoMPemuQ3t9h5nwWPQStOVDTrtJt55hIm40OnQbDfAhTk6jkPG+WFuQBR0mjFJfPqb5RynBu8A3xRCfIPDKFO89//dw77kvf9bwN8CqCxb//lpEa2jmLoJ70piKjzeOexUKF+lhXDcbd73BNOntQZjLFR+ZyFAngZ7FgLpFUopJq5kkpfoCJaWqkg7oLCBHKZFiExUEsYZQdeQBBfizrggktBJQGiBlBEoSTPSxEpjnaxcr/VAKUFamJnqjhLCxroLrfoy4y/2uszWVneIbq+WYgAQQlFHzmkFIZT7A8A6z7ff2eGFF+/wG6+uM8oPLUntWPHzz1zg+auX+YknVw5EwpuruhYLRoPg+qkLMTENyplHI7YAW9Y3bhTWUphAImaxUUMHaDRgOMmJaow+V1ITi5DmQKmgG1sxIY/ZlHgpYCqOiaq/U8tXyEr1XvjqWJyEFEKjPEOKGuPYhCDWkknuKSzoWNEiZn+cE5kcJQSuirzRStJuJngnsZRIBDIChCZpxUipKYqCsgw5L7W0JJEms5YoLz82ZOtm9YqZrYRm5rjvlj0SvPNYH/4XyJBDS0g0vvLFVd8lpArwlTtOikqkbsOO5FFNmqaj8NgQDm89UaRQyqF0sNoYB9kEhhNIRHj4FCBkSGoq4pAsuLAhgnFiILaedluz1O7iURgPOIOpUUjuhaDXSkji/MEb435IjIELK9BOmrWR5EZU3+NTQxLoA2gVrMazNpe0FEh1/IZ573ltbcALL97h1763ysbg0HqkpeBnPnWW569e5mefPk8zfi9dLL2rhYhPgKRGbVOzHSNnQkfuRo9686J558iy2Vi1pigmsDNJqXOoTyLJIM0xxiIkZCb0GUtIiNum0l4RJuhm9XdCiAjd4f5JPVOCmHrxLFgr2Bll1Bm456zDWI8UkkhLTCrJ8hxjDM5a4jhCo/DCorSmGevqs4RB6ZG2QCsRovAFmDLkCjPOE8caISWRVGTO0b5nW77TwHFSP/ydD1uI9/6bwDc/7HU+LO63ZY8QAus5cJVNIxN9xah8xbem5Crk3BKVhctVIvWwz2Ld6R+8c5TGk1qBUgnL7SUG6RZm4mm2glZrZw+aTcjTkNg06oYHzAKtJBw7EytkpBHOURSGSGtiHZNEGuMdhau3XQrNxaVlGp01oslsIhLPAYmOkLq+0OBY1ZjgscZxRQlY6jDbWQ7otI+Xaf323oSvv7TK11+6w5sbdwvdrz66yNe+cJm/8CMXWek8PCbPO1eLZSsFIlmfu20xSWpJadEhWJvqglaKtAikI+GIG+YEMF1LjMtgWarR0I/wjrEzxDpmeaFgZxOGJrRxmktLVf9P00E0FSgbrKY5QcOmCJasfYL1awU4uwQtHWGFQIuQk7EueMDiMT7kzwpSn4QkDo5PAUSRRGlFrDRaS5I4pp1out6z3R/jjcN5A5WuywGRkCih8R5K54krSdBp27YeWL4Q4h947/8TIcQ/5T7E2Hv/CzOt2QxwdMueKY6mgbj3HE9YAVsbBsKwoaUk5J8Pps1potPTSf0AzUZMrxVRFDm5hIVGjCJHSoUWnnNLDhkJ8onHy3AjG01BIjSdRhOpJN0kQukICQyzjEhpIinpJDHNJAoahRrN5p1OzHK7RacJZ+HENTM94LEFaLWbRFLXRpKlljxKEEHOGu36tm5DqohLF6HbD4P6LLAA9JZ5YE6L/UnBP3tlnRdevMMfvrt717ErZ9o8/9xlnr96icdXji//TaKEpRazS/RW4bwg5FypCY1mi4vMvh92FPRanRmXcohGEtNrwrkejPsnt2XPMoE49hQs9mCl06UwNUYjOrjY7uKdIjd9Lj9akmzBxihYtUoCEVRU6Sli6LbCMyMjyLMQja4E5Dnsm0C6PnEGzpyBXqdDu6FYaDbwvs4ALxFSKuHxzpPEEVlu0FLiRfhcRxGdpibWCi0lSWUUEQi6rQbWWUapQCmN1oI0szSSiCiSeB+sXa3W6Vq0pngY2fs/q79/v46K1IGjaR+OaraUFHcJ5I31eO8Oco5IKauIwyoZKryHsJ1KtIMQtBsxvU4bh6aVFUgEO3IUOqeqMvFKiV0BhcRLRWlKIilRMqwoFtotmlpjnKEVa6SO6MQNet0mQjiajaTW7Sl6jQa9xTZXLiVs7uWkg5Av5iSwAjzRgKc/k3C20yOpMZnkYrvDlTbcmrEH5xxwfnl5toUcQRJFPHV5mWfe3eXF7GQtChAGqaeAK490aDcOs7VmpeW3X9/khRfv8C/fuFvovtKO+WoldP/8I70PRKh73TaPXIInrzGzjZufBD79KYji+jZYajdbfOop2LwG786ojAXg4kV49Gx9Wxl0m00unGuT+zFFAVEK67xXrxRVr4KgUpjmMrof2gSytdyCXgceu9ig1+jVlo8KwDvJueUlvG6Q5QVKa6ROUVvQWQJcELlHUZCNJBF0FyDRVcqftMrx52DBwZIJ7e60YLEZ0Y4VS+0OC51mrVKYabS/1opIa5R0FIliYEqUkLQSTRwrkkijZdB3SRE8SlJK2olmkgtaDdA6IlIKKUu8c0E87x1JrFFSHexzfJp44Azqvf9u9e9z3vt/ePSYEOKXgG/NsmKzwDQicZqEVEAVhXY3CVPSY23QNk3PEQfE6wGE7RR2FA8dULHQjpHAQAmcayClIVGabqOBlIpBXobtG1SEdZZcS5T3WGCplbDY6dJIFOO0wNogLGw2YpCCVqxZbEYkUX1kq9lMuNTr8tlHLzHOrsObsDkOg+Mmwe1ymCntcLA8KhuahkdPTec5IQLxqQZ86gk4312g20lYaNRnUTi/1OULn4LXXwztmBWeW4RPXjo3wxLuRq+dsLK4yLOf2oU34Lv5ybl+FwiE5HOfhqfOrLDYbvL7b28fCN2H2aGor3VE6P6TR4TuHxQXFzs89UiHra0R4z4nuk+nIGw6+6lz8Oj5LhcW68sLcnahxZOP9Nge9Mk3Ye2Er98EPqvgR660eOxcfaR/sdPkqfMrDLKCLC9pjqC3D3s+9KMhgWQ82oRIw/4wjA2KQ12TIiQBVYQxRAMrC7DShTM9uNRdYGmpQatRnyiy01TsjSXL7QbZ0iJ6NMGanM4jjjiSZEXYi63TkLhlgXYe3YiJVMiPOGmWZEXO/hjiKDQqttBoKS73Fuh121xcbNFtNWjUKCRXSpJohXWOdiwYZdBrxyw0g8TfS0E7Cjm2lJYkSmK9IDgLRbCAAbr0GO8QwrHQUAipUULQagQiGel6PTMPwnF+2X8f+If3fPZX7vPZxwKiyqt1L46SMCkEOr5/uoMHEbbT2K4nihQNE9FJHAKBVpJeO+Yxvxy2PhCS3Di6WYaVkjItSZ1jqdUiSSLiak9HpSStSLPUjgFPacMqIo40jUiRRBFRVJ8pVkrJcrfJJy+dByJWFta4szdkaweemBD2o9QhylJ6GBUwngQyFhH0CgsLIW/YOA2rPgv0unDpEjyyuMRSt8OjKz16nfosCisLbb74mU+yuv8Wv3edmWQn//EY/vSPLnHlwtkZXP3+WGg1uXJukf74AkJt0r3leHs7hDHfC0VwyUw3xU0Ik3MHaErwLrh/BCEP3LkePPq4ZKW3wj+7lvLL/+JlNoaHtjMlBT/9yTN87eplfvaz52md4GSxstjluSufIM3eJLle8NYGvPk+35lS96P71nWq/y3BI1kAnwCe+AQ8cTHhyUuXuLRSnwXoTK/Dpx+9jHGOxd6Qa2/By3z4ranbwAXgyWV47jMNfuaZT9Lr1udG7LYaPHXpAoWDhljlpjK0Eng8AuXBSmhqOLscE2nHxp5BAottSX/suLUe5oFODKM0aHUXlmChCa1ui0e6XS6fX+biYpdWUt8irddu0k8t3qec63XAWjQLCCXBhug7pUJuqoU4RitP4TxSaRIpGRWG/fGEs50UIcEjaTVjznUWaLWatOOY80ttznSbRDUuqkW1GwuAkpIkcnhccHlKTyTDdnlaSpI4pOjJS4cSHkdI89BOJFYrstLSihUCgVSCWCriWCGFJI5OaTu9eyAepDUSQvzbwF8Cfgr4nSOHuoD13n/5pCvzpS99yX/nO995/xPnOICrRO1ZaShKg/OEzuUhM46isEjlcU5gnMWUDqUFiQ5ESmlFpIJ5Vlb5kpz3B9sWRUoSRaq2hKZH2zXJDKN0wurugNXNPoNiQiwFCoXUkCiFsZZBmpGXBiGgkyTEOgwwyouQysJ7EBLnHe24ybnlHpeWuywtdGk24toeRO89o3HKtdt3+Pb33+blNyxbVSaIikMiqbY54ZCIOGAhCiHapgxpPAofVuA5YUV+WcLnnoYf/cxlPv/EJziztFDbPfPek2YFm3v7vL22x+b+HtujAZORYTAKm92WHpoRLC/CuaUuIEitoRs3qhW4xRoB2pM4z8g5tvYLXtvzfH/Lcntwt63suUcXef65S3zl2UuceR+h+4dBlmXc3Nzl+toua/19+v0+g0m1t2gSonylhElZBaFUGwX32nD5TIyXEi8kygomrkSJkN26FSd0OgtcObPAExfP0m7XlUoy3K/haMKtrT1Wdwds9/fZ3u/jXJjoygysApdCPw27S3gbcg/mAB7OLkCnUyVC9qFfNtqwsrDIJ86tcOXiGS4s94iiGvNReU+Wl/SHI1b3R6xt7DEwGW0ZISJJQ2uSSOGcIk40nUQhvGc/d9isYFyOKUpHrBPajeAeN0aR+5JERqwsdjnXa9FptUji+rSeAEVR0B+n9McF4ywnUh7vJMMsJSsszVjRaSUsNFu0mzF4yMqS/UlBmhUhZYRUWCGIVfCIeARKKhaaMe1mQhTV26YppnsUG+sOMgZIIZBKHspzjqRZ8j7MUc4eiRaeRrIRnj+pZLiGnL0hRAjxXe/9l973vIeQrceBK8B/A/zNI4eGwMve+xMPyp+TrQ8P70OSVR/ehI7s/BGRvzjoyHV1xg8K5xxlaSmMpSjLECkqg+m5EesQEVq1zXsPIsSwKCWR04cSwHmUloc6i+rBPY22TweWojSkRUGWGxyeSEiiKOzhlZWGNCswODSKRkMRS43UCj8dYESIgFVTQiUEWiqSSBFXv81ptSsvDMY7qDZ/l1KipEQQVuEIWW00K7AOrPcoIYgjxTA3fOPlNX7t5bX3ZHR/fKXF165e5hefu8yVM/WRk+mCJi8t1jmE8FWslAhSAw+FKRlnJcZDogSdZkISx3jnKKylNB4hPFoIVKRRVbi71urUJjhjLKUJ/UlOty+rcg3KSksjBRRFyTArQu47PK1GTCtJgh6m2neuNPYgkvu0FmjTdjnnw4KxqtN0EaOkQFR1EoRkvL461/swHujK0j+ViEyv5Z0/2LLttMdM5xzGHLZLa/me3/po3aekRIgjUfOnOAb+sOFDk63TwJxszTHHnzxkpeWbb4SM7r/9+t1C9+V2zC88e4nnr17m2Q8odJ9jjjnmmBWOS7aOsxH1jwH/I/A0QXqggLH3vsZUs3PMMccPE5zz/MH1Xb7+0h2+8craXUL3ZqT4uWfO8/zVy/zUU2dqzdY9xxxzzDELHEcN9z8B/xbwK8CXgL9MiMSeY4455viB8Pr6gBdeDAlH1/qHQflSwE9/8izPX73Ez332Au3ktFMQzjHHHHOcHI41onnvrwkhlPfeAv+bEOL3Z1yvOeaY44cEa/2Ur7+0ygsv3uH19bvTnn7+kR5fu3qZr3z+Eme79eU8m2OOOeaoE8chWxMhRAy8JIT4e4S0LPWpU+eYY46PHQZZyW++ss6vvniHb1/f4ag09PHlFr949TLPP3eJJ87WlxpgjjnmmOO0cByy9e8Rgh7+Y+A/BR4F/s1ZVmqOOeb4+KEwjm++sckLL93h//3+5l1bmiy1Ir5aCd2vPro4F7rPMcccf6JwnI2ob1T/ZsDfARBC/CRwbYb1mmOOOT4GcM7znRt7vPDSHb7x8hr99DAfViOS/NxnL/C1q5f5qU/Ohe5zzDHHn1w8bCNqBfxF4DLwm977V4UQXwH+S0Kuxav1VHGOOeb4qOGtjSG/+uIdvv7SKnf204PPpYCffCpkdP+5Zy7QmQvd55hjjjkeatn6Xwguwz8E/gchxA3gx4G/6b1/oY7KzTHHHB8dbAwyfu2lVX71xTu8tja469jnL/f4xauX+eqzFznXrW+fvznmmGOOjwMeRra+BHzee++EEA1gG3jKe3+S+7HOMcccH2EMs5LffHWdF166w++/fbfQ/dGlJs9XGd2fOjcXus8xxxxzPAgPI1uF994BeO8zIcSbc6I1xxw//CiM41tvbgWh+2sb5PcI3b/y+Us8f/USX3hsaS50n2OOOeY4Bh5Gtj4jhHi5+l8AT1bvBeC995+fee3mmGOOWuC957uV0P3Xv7fG/hGhe6IlP/vZ83zt6mV++pNnifVc6D7HHHPM8YPgYWTr6dpqMcccc5wKrm0ODzK639q7W+j+E0+e4fmrl/n5Z87TbUSnWMs55phjjo83Hki2vPc3qojE3/Lef7nGOs0xxxwzxOYg49e+FzK6v7p6t9D9mUsLfO3qZb767CXOL8yF7nPMMcccJ4GHxmV7760QYiKE6Hnv+3VVao455jhZjHLDb766ztdfusPvXdvGHRG6P7LU5PnnLvP81Us8da57epWcY4455vghxXGS4GTAK0KIfw6Mpx967//GzGo1xxxzfGiU1vGv3tzihZdW+eevrZOVh0L3XjPiK5+/yNeuXuaLj8+F7nPMMcccs8RxyNY3qtcPHbz3WOsw1uGcR0qBVhKl5Mdq8inLkuEkYzguKVyB8B6tI2KtaMQaJSSOkO3bWoP1AAIpHNY60tJijENJjxAhGq20jkYcs9Ru0Os0ieO49jbtD8fsDnMKZ0gkKClIS0tpAAom45TttKDISzrNiLO9Lq0owSowhcf6EufAOfBYXOkopKepmpxfbnF+sUur1aq9XYNxyv4oZ1JklEVBXhgy54i8pNGURCpCCEVZZIzynEnuca5AYimtxCLoxhKlJeNJwc5kQqQ153s9nrq4wnqq+PVXN/n1l1fZm9wtdP/y0+d5/upl/vVPnazQ3XuPMZa8MOTGYIzFWMMkzRkWBemkAOnQUhNHCpwjsxZbgoqgHccoKSmdI88NaZnhvaLVSDi/2OZsb4FmM0HKesX5xhhGk4ytvSG3drbZ2h9RloZIO2KZEDcaJIrquYG0zBiPxgyznKz02BykhngqeROgI8lCq8vj58/w5IUznFnqEUX1auKm7doZjNkdjpmkGYUzRDIi1gJrSyalJy8MWjmaOgEpcM5inUApSbuZ0GnESASF95SFDW1rtljqNum2GrW3y1pLUViM93hrcd5RWo/zHi0FcaTRSiGFQEiBs47ShPOFByVBCAlCILwHAc6DEIJISaJI1d4HTwLee5zzeEKEmxDgPQfvpRQfqznv44jjbNfzfwghmsBj3vs3aqhTLfDeU5qKaFUPk3FgnUUah9YSKcRHvhOmacrbm/tsbA0ZZim7ozElsNJo0kkiCmfIrKMhBdZ7EIJGFCaIvTRjlGcILynKnP1JBli0jjnf6dDtLLKyEHN2scdjZzq1Ea6yLLmzM+LOZp9+OuLW9h7XVwfsD2CpAwsLsLsLgxSshUkKwwwkayy1oNeDOIaihFiDSiAdQD+DZgxnzsNTyx2eeOxxvvjEudoIV1mW3Nzsc+3WGm9vbHJ9LWNvAImGdhMmJWSTUEchYH8CLoeoCfkIdi0sCIgVTAykhB3hdQK6C3t+nzvjm+xnhz5CIeDHn1jh+auX+bOfu8DCDITu3nvSrGBrf8jNrV1ubGyzubvH3jgM6EpAZqEswDgQFkoFnTgcH45hPAAvYKEBhQDlobkIF5bg7NIKT186w6cvn2ep165tsjPGcGN9hz98/S3+4PU+N2+DJUxOI8AxIoKwkCFsqzEEdgFDOLe475Udbfo8RZ8/9SNv82e/8Bl+5KnHaiMmxhjWdoe8u7rNta1dNna2Wd+CNIXSQj6BtIROGxCwP4KYQETiCBpd6HUgkpDlgWQ6F+7z0hI8dqHHU+dXePziOR49s1Bbu6y19McFRVkwygz7owmZcSw0NM5LMlMivCdSAiUFiEC2nJA0tcYBxjrazZhGpCnKkklp0Qh0pGnHik6zQacVf6wIl/ceY8NCWgiBc46y9ERaIKU8OK4VpzrX3UsIP+pz7w+K9yVbQoivAn+f8LxdEUI8B/zX3vtfmHXlZoHpDTU2WHWc9zgfVi/OObz3JJFCeYkXHHRC4KAjHGR2FOJUO4Uxhne3+txa22NnnHJza4v1vsVngNij0QAroReFyS0bg9DhfebBGigsaHk44U1KaMQZZy5kXFmZMMgWEECvqThbE9kapwXbu/vc2Nnl7fV1rl2HlwewByxuhYltKuuOgYQwyQEwAT0JK7Y24e+Y8B2AJ4DRGPb6I0p5m3Ndxacfr4ds7Q/HfP/d2/z+9TvcfBveTGHzyHFJGGRcVe8DHJ2xPWEmP4q8eh2cAJ882+Iv/ujjfPXZS1zozVboXpaGzf0R37+5zss3V7l5p+TmesiCnBJIRwxMqmpO702r+t8D+9NmZdAAekB3H26vwaMXdxiPBkSR5pkkotWqR7i/s9fnd75/jW/9QZ+3RrABlO/7reNhDHwP2HoFJunrnOm1eOzSxRO6+sMxHKe8u7bDS+9e543Vkhs34Q3uQwzH97x3HPS1ePu950fAI5vw6Hqfzcf6FMbTTRRnlxdn05B7kGYFo6xgkpfsj3P6gwnbwwneFyidoIPBCiXAeklpCiZFQUMndBsJCIiUpKEjEJ5xXmIcLC+0SKKEvoLl0qIltfXBo3gQGXk/kuLcIdEK1wEpj05jAgjXUOp0yM29hPCjQgBPEsdxI/5XwL8GfBPAe/+SEOLKDOs0Mxy9oQiBA9LCIWUwizugKB0egRCWJNYIQUXKwvc9YEz4Thyp8P6UOsUkLdjZT+mnGTe2t3nnpqU/hFEZxkRNmOQ6jTDATDIQHrSG1ITJTxNW5GPC4DkEzhaQ3YDCTCjxtOMmy92Ms0u9WtqVFpbN4YTV/T5vXYdXKqIFYVI+ioL3DvpTLjLgvXgHGE/gzAR6rT6fWBny6cdPsPIPwcb+iNfXNrjxNryawr0RJ+6+3zo+JHBRw7OPwX/405/gi0/X85hmueH25jZ/dO0Gb1yDmync4eHtGVWv+16vem0A7RzKm1BMSs4vbvHISq+2ie6d9R2+9+Y+74zg9ozKWAVevgZ/fGO1NrK13R/zzuY2L79V8vZOeCZ+UNzPYlcC1wH6oFdhqbvJpZVubWRrkhuMdQwnOXuDjO3RhJ3xiH5/QLPZJJKeSVHgvKeVRDgvyAtDksT4vSClaDYTOpGgdJDmBYVzLO91ObfYoxVFGGtItKTZTGod7+9rnSocUlZWRfVgK5Xn7rlp+t5YdxdB8/cruCbcSwinBNBaF8gXH39r13HIlvHe9+9p4Gnelw8M58I62jkqrZYNWh4XTMJpbpHC473DOVl12mmnDC5F73zw9ftwDa0Vp7UqSEtLaUtu7+2zsWcYjOBOCWv3nNfKYJkwueWAMXcbSCSHFgYIg2ZiYGsduo2UpVafSdauo0kBwrGxs8uNWymvHyFaJ4UNArlceBO+cGX4fqefGPqjCavbhtv3IVonAQd4A2oC2+P62jVKU/54fYfVLdhO4dYJXnsMrDmwm3Bza5ud/gUunlk6wRIejNXdfda2YNbbZrwDvHFrnT/34zMuqML2cMjbazu8tQPvzuD614HODtzeCnqwumCdY5TmbO2P2BtnrO7tsjPIMCXkLmVswIyBCLrtgtKAK6CUKS0BcUOy2x8zLoJr32soUths7jHJUxYXl3BYWknE2UVDHNenRztKRrz3WAceT1E6EAJjQSuHqExWFlHNTYAPnpypJ8a7oGNTRyxjpXFEp2TVgvcSwilKG9ydPwzWruOQrVeFEH8JUEKITwJ/A/j92VZrNnBVJwWP956ssKSFQQhBotWBxcv6IJ48sGo5j9JBNO+8w/twrcK4AzG99/XzT+8deVEwnqRsb8Gt4v4Tw6R6PQj3WiBGwC0Pyxl0N+Di4gQp6+vcGtibjHht9W4320liBFz3UJTZjEp4L4y1DAdwY4ZlrALPFODK+vrjIEtJh336e8EdddLYJPTRnaFjkNZ3vyZ5znp2xEM7I4yA/v3MsDPCcDThzupsiNYUW8D+EKy51+c9Oyjp2R5M2BuOGJWOjd2M3X0gD6Q9qh4JCUwmYLOwsPQ2aARbHcdwGI6XNujWJNBqh3HiMbNDQy5xwS6Q5mWtZMs6h7Ue6z3W2opsSJz3xJEiLwyF8cEbQ9CeAQeeGE+wfnmgNMFTE32EdoKotqW5i0RZ65D3sXY9zLDxUdZ9HYds/XXgbxPGnH8M/Bbwd2dZqVnBO49zjqJ0FCFEDbzFGA/CEyuFkAKFOzRfOo+q+mSIXgwRKtNjxnqU9MhTuKGxklgvcAYG45Ndge8TtDOjIeR5gTrBa78fclPivTtxi9a9WAKSOJlxKYdQwmPur5g+UZgS2s36JoIscwwzx/oMWckuMBiAKeubvLtJRF32QX2ckfiEkJcF6Yz74TogHURxfSOHRCK9Y+JKhnkZhncDq6OgHewRSPsIiMfh/zZBv4SDtICRhU4UiFZOcJfGBQxHsNsq6I0zdBWxXRecc+SFA+HxyBDI5T1aGKzzFKXBeoGSEFUWsNI4lAyuRQ4lQDIAACAASURBVCFFNdeFACkpBZEUyMpIIDh94iWlwNjgXzm03gXrm7HuPRq1++Gjrvs6TjTiBPjbQoj/Nrz1xxp/hBAN4F8R9Msa+Cfe+1/+MJX90BCQF47CWZRUCC1whcIJh/ICJWVYFQDKOaQLZlfrwPmwmlAqdArnXBU677EWdFx/Z9VaIbGMR+/Vsp7I9YG4FX6PvA6WUGF/nKOEQmBnWk5PQqTrS2nhwmJ05igyaMX1zd7OlwyHD7eefugygJ1tcK4+stVqJdTROyKgVWNmFeMFugbDp4qhWePzhZI0mzELUcS+GZIOYNQP5Go6ehkOJROC0GeVA0WIkh0RopglYfzThKChThlci1LBuDDUaOjHGIfWIqS8ER6tFGVekBEE/YV1KCFwTpCXNqSo0DJE2XOo8bIetArpjdw9hMX70zEYTCGEQKsqCM17qEigB+Q9GjUlBP4+FqsH6b5OU/h/FMeJRvxR4H8FutX7PvAfeO+/+z5fzYE/7b0fCSEi4HeFEL/hvf/2h630B4H3Hmsc1tkDcZIUIoQBC4HUEuuCRiuRAhAIUXVa5yhLhxBBu2WtDfovL5BChvxcVtSen6ssDYV1mCgM2CcNAyQJLLbajFL3HjPvrDDOCnb6duaWrb6DSM6W0N0FI7A1cAVvwsq2LjhrGduTi9R7EMYOSlPf/YpENJPn6l40gU67vsXaQiP50MEYx4FwIGR97mzpQSJQKiYvSnIfFqE5YSybkqcOgWS1CONmxmFAUZNgRfUcRjovEtKSFL5Kd5GVtW7G7gClFM7bIGEhJMgy1vD/s/dmz5Zd933f57eGPZzhjj3iNkGABElxEImmaMmlwWV5lCw7AviSqlTlwUmVypWHOJU8JM4/kOQlVX7IQ5RSqZwqv6jKJGhJtmx5kCLZMRURACeRlEiQBHDRjR7vvWfawxrysPbpvj0BDeCe3U3ifKtu375n3Ovsfdb6rt/v+/v+MlNAcKBThE5IhEMtyw1jpO2iPSm9FvDOE4E8S5GvEALBQ5E9WkKyDGbAUg+tUqAjhFsatRAEY+W+EasQY+cdFh8qEtY3Hmb7+xvAfxNj/CMAEfl54DeBT7/Vk2Ia4bLgyHY/j2TUy3AiSlBaE0Poyl4jxhiIYLVQZIbWeXyIZEawnU5La51EhG3oLvx0gS7aiNWBQunOOC/cek4fWKStDiWriSoUJJ+qTGuUlt52CM7VvHYl7S5XuSg0QO36I5F1bHl9leGfDgcRrhz1JwJqAgzCvZWiJ40I1L4/EtmG9gE+WSeLANgeTYPzrIuSrBgxJIFzX7BWIxKY1TNuzju5B110hNtB5WV6UEgRrWWi85B0LjZJRTmL7jkNIApygYDC2H431YpEOJQIURKhVFpQIVUf2i5QQBREK3wI4CC3aWTL9U934bjWh0S+uu+SEkGb21YQjwMiiTCKpHUXSJmnjkgutVnL9ei4xOd4VeajkvjcDw9DtiZLogUQY/xjEXnYVKIGvgI8A/wfMcYv3+cxvwb8GsCTTz75UAf9TrEML2qlyHSgjoILgRiTAM+5ADoJ5DOjab0ns+aYL8kxq90IddNFtkRofcBohTHccwGsGs5Hqrrm2ooKfgQ4OISDasZGkfXGlIOPRHevndRJYwOo3fGq0pNHjJGvvn7ICy/t88++st+LBmgCXD3ssZVpBGeS/mWV43NA6DGyNVtUK78GIUVPhrY/36YYhNDDx+gEXNsfOU6LsUoC90UyPvakaFVF2gwEYLe7rSEVX2hS2mbU3TYlbTQhLZBDYDSGUV4wGozYyrNOX9QPjFFUTUAkEqPgnEOJYpgLsas6TLYXDqUMWiKtSylDkBTtInmM+RAxRqe1LVWEPXYRILhTMK+0SqnDLrUI3HO8y3XXh9vPizE8MonP/fAwZOtPROT/JInjI/CfA38gIp8FiDG++KAnxhg98KyIbAFfFJFPxRi/cddjfh34dYDPfe5zKznby7JSpRLr1ZIE8dO6TQJ4a1DSmb3J0vTU33q295HGe6Qzg3Mh4mNqf3JL0xUiWqteQ3dKAtdnE1yzGmLiun9c01Lktretj4tQZKxGiHYMLeBDmyagE37tH1yb8cLL+7zw0j4/uN5DOOsYrgOzWX8au9wqvHu4yeS9YAioHru/TKaLlaeyoUtx5f0NbFKlNlarRjOHNvR3HcaY9D4hphZJEm+7+9ekSJUmEa+G22nEOYlg7XT3ZySS5khzRAW4BlQIjKxB5xl9iraUUhRZFxQgogQGuU6tyGLEeSgygw9dFEcUhRUESfeHSJGZpI/vbB5CiAgKpZIVUkpBPj64QzBPiuyB3BHJOn68xyNhS91Xasn0eIjj4eHmx2e733eL23+WNMa/9nYvEGM8EJE/AH4J+MbbPPzEsWTJ0PW4MgrfeJQodKbIjSYAk0VNbhTDwuKD4LzHh1RSrBB89EznPoVwO8d5a/Stigmt0gXeF7Qo6PoermI5r0gTVZFpENPfzicGdA/6bgfkJ/hG16c1v/O1S3zxpX1efu3OpNrHzo75zBnDq6/e5D+tOOjkoZc00RJllqHs6ifrs2egNP35vV2fLFZcopGQA1mP88bBfNJLelQUSOwvquBDpHEtN2dTggfR0IZEqI7rCVsSqfLd70F32yUSobd0wvLufqtS5406eqZ1S1s3ZD2361FKkXURGm/0rU190zhcCCiELFNopW6JypdVh7FOvUoD0LpEYJQIkc6vSwIi+rEQkS9xXDAvAqTk062IVozccbzHI2EPImSPGg9TjfiL7+aFReQ00HZEqwT+BvC/vZvXeq9YsuQQkvO7UhppPeNBjvMRHwJKayQ6Wu9BcoxO5p+BtBMQkv9WJGkRfIS6cmwM1a0dhHQXeI8DY2AVElZX5FY7cI3H6GTk2geyzGLy1Wu2cqDI8ltahneDeeP4/T97kxde2uf/+YtrSTvR4exGznMX93ju2T0+fn6DP/mzP+c3X199rKQBVpQVvS+yLOPMIEUOVoUNoFCwvVmu8F3uxGKVAzqGAmh7oXUJEqSPoliip0tV9YOqqZnMA8O8RGc1qivaWJIquv+r7u+WFNHaJC3WDSl6kHePmZPSi7ZrgFk7z9H0iIUfkec9enXcBRFo29CRpZhSbCSC5brolo8hVdyLkFnFrHLEELr0YcrOaAUheJQIJnt8/KiWWBInDRh9O2IlcIeQPsmvu4imejAhe9R4mGrEf0gSxE+A/wv4LPA/xRj/9ds89TzwTzrdlgJ+K8b4O+/xeN8Vliy56VZuAXTnAh9JonmjFY3SuNZRVS2oVEY7yA0RldKESlFYxbzxKIE8i0gMxCgURvUqjgdSeaxojEm7s5MUKI9JPlRKw40J1M2qrR1vY1gUZKyWaNG9vo8Ord/Z0uN84D987zovvLTPv/rmZebN7YVylBt+5SfP89zFPX7m6Z07yHduM7IBJ2+Jfxc2gPGoP/+w4FqmXQRhVfxkCCw8bJX9Ccn7ci1wQN30F4o0th8mfuUoRSP6wrxqMQryzGI0qAzKKgnfl9/QJalapghTVDHddprOW5A0N5jufgLkJWTWkpU5rlWPyMQ6ta9pfYQYOkuieKu/7zLFWLUt3gNFxEadbIq00CLULjXizq3uZCEKbR4/onU3jkes4D6eWulGiLcNTbV+vMb1MPT8v4ox/mMR+dvAGeDvk8jXW5KtGOPXgIvv/RBPBolwqVv6LaOEw3mDKIVS0DQt86YlxICuhTzPCD4wrVrKLNXVSowgitIm0Z6SRNrKzJBZ3fuJFYBo0PHk5U3L3eD2BngNh7P6PUWA3glGecZi1T4CpPHFzrz27RBj5Ov7h3zxpX1++6tvcG16OxFjtPDXPnaG5y/u8Ys/cYbiAYuZ96GXiIIFtof9pdtmTYOEtFCtqgYyA0YlxF4+wYSdjVXHVhMOgHnV32am0P30wZsDRvUXAWpDxGiLVoYsS9FdTdoAWNL12QWpcCSCtUUSyzvSYwtuN7Yvu9u0gukUzmw6Cm0pc0VVO0Y9OtEuyUUIEaUE71OEKs8MdeNRKhGNed3Seiiz5LNVt56mDQxLg0GIJGmM6+aizKYqxh813M9Ta6k7e6eb577wMFfL8kz8HeA3Y4xflceJLr4DLNOJMXY9pADvPU2b9FiiBJyi8hEbI8YaFrUjN6nisOksI8rC3DKxMEpu9ZjqGyFGisJgx3DhZppUTkKLkZNY9eYgGRPuDhSua03UB4rColSKrK0qCHQWOLML1dtUFb16fX5L6P7KXWWfP/3UDs9d3OPv/OQ5th7ClbJqW2qfJvhV2iQ8CWwMByt8hztRNbC9aTiNW0l7pV3gXA57pw1HVX+C653RBqs3tEgRlrYPA7YOWV5SLkVLK0QO6B7z2UYlX68is2wNSrZ3F0kUPr9t9WC4vSmwpBTiqTMwmifZSJEnU+BcpYpGUTDeTJEt7wVRBqUstfOMehvZMe2SyK1KwmVRlzEq9fqNES2KcpDE8K2PWAV2aesgEGISxavu+T5Ir5KDk8L9eik+bhWVd+NhyNZXRORfA08D/0hExvSx3VsBlunEtk3arY3SMqkalBIyZZJZXZ6hJOKcx1rNqDQQOh+mAGWmMVp3aUV5pG0OIsL2YMSF3RtU85bJNXj1Xb7WOdLkeB04BXzoXJp4Noewu7lBWZreCKVRltMbcPpSCumvIsi1Z2DzFBTFvSTpxqzhd7/2Bl98aZ8XX71zsX3mzIjnL+7xq88+wYXtd0ZoYlTsbMCZy6tbwneAZ56GYdkf2coyxagoOX9qwuE1eP0EX1tI6Z1zZ+H01k6vJfej0QY/xQFv5978XvEMMCr7W7rLrGDvPDz5+rufL94OG8DuCArbX5XlqCgwNimvdsebVK2jLFvGN+FoAsHD1k5Hpmaw1cKp0zAcwniQqsppQcYwLFL7nnoGgxFsjjSbw5InNjewWhH68M44hiW5EG7rlo474UuXZVHW3PKF1J2h7C3JTExGoDFK8pLMun7Ajy8/eSCOC+KXeNwE8XfjYcjWf02qSHwlxjgXkV1SKvFHEsd9O4xWpKbpKaxed20Yijzrmn0qlES0VRSZ5lbwPaa89zLS86hOcGYMWxsjnj53mtq/wWwGskgahZpEUnZJBGqHRKaWYfRA2tgu/WTOb8DAwtWbyci01DAu4UNntjm3s8upUX+LQZYZPvzEJt9/8xA5SI2bT7L98MeAnW3Y28rZ29oCYNF4fv9bndD9z6/ijgndz4xzfvXZJ3ju4h6fOL/xrknnaFBwfnfI2eGMySxVP50kRsCnxvCpp3bY7vF8ndkYMh6M2B5PuDCBG/XJVceeBc5vw4VTOVvjERtlf1q0ndGYn/oM/PCrcG1F7zEEfuIDcPbU1ore4V6c2hpy9ozh7GVHcCdLjiHNNWeAzzwDm6P+SP+wyDm7kbNYlAQnnNvepMxnbJcNN488QSC3SUR9eAQ2gzMbmuGgZLqoCDEysslSQeclVgSHR+ucMtOcH48YDAqUsdg+m1lym1wsszMi4H1EiJ2JZ3qckts2CUYrnPOdabfC+US6lCSipXV6njyeWbe3xP16KT5ugvi78TDViEFELgD/RbfI/GGM8bdXfmQrxHFWnBuVFlYRNgpLE8A5l/pLaXA+pQqVCIPC4gN3NLp8lCd4WFg2yowLu1vEqNDhdcrXoV5A1Kl9RevhqRbsEKxO/jPDAbQtEKEcgpWUti8sPHMBdJ6jteHs5gbnt0ZsDYfsbPSnASqt4UN753jm+iHGQH4N/pyHJ1xjutQMKRU5Im3eKhL5vHAGPnQenjp9hksL4b//rZf5vW/cKXQf5ppf/tR5nr+4x1/+0O6J6NXObg556twub1yf4X4AsTm55uEfBD6yDRc/VvDBs6c5tdnfIre7MeZTF04xXRzRxhl6H/6svX8KeAu4QNoMVNzpc6RI52xM+o4WwNND+PCTig+f22FnPOD0Zn/X4fndER8+f4ZfuH6FF19PpP8kMQZ+egc++cwWHzm7e8Kv/mCc29zgUx84y5Ub+4TXU7eXNx7w2KX55/E4zoh03jRpY7eMPBtSFPIM8KEn4ac++jRbw/6qR/PcMCpLnjjlyWzOblVyYzBiOjuizGdIGxhtjCm04mhrQaE1G5sbZJLho8N5z0aekVnhyuEMF2FcFhhRoAybw4wyE3YGmrxnsnWcZBkteJ/E8Fo6N3lU56mVUo7GHDcB7WwgRKNCIITOyJREPB8Xh/V3grt7KT6Ogvi78TDViP8r8JeAf9rd9N+KyM/GGP/RSo9shTjOiq3VuNojwKDM0Y2j9antjhYhy/Ud4vfjpmmP+gSXRcbuxgAR6XQKhrPbN5nXNaW1hCj40HbVloKKkTpE6ralzHJyK7ReszMqsEqjCHhtGSphOBqwNRyitWJ7mDPoMaKQ55ZnnjjNwaxiOHiTvdMVHziA6QF4n3akNoe8ywB6l0L+xkMski/OeAhbBcxaOiEpzJtUXZkVA7471fzWv7vMjflrt97XKOGvfuw0z1+8wF//+IOF7u8WO5sjPnz+LNE7Njaucu7NlskUFhW0dSoxt5LEuuJA8rRTVZKa41ZV2oVulHAwS9HJgU2pkSd24akLZ/jgqW32Tu/0GlEoioynzp9mWFr2dq7x2oVrfOzqnDevw9E8RRMCqVdeOYazO5Cp1Jg7tlBFaJs0Pt+mjcIoh9EQTu+OefrUGU5vjzi1NWKjx3Gd2hzzsQtPEEJge/saf/F9uDpN4ulurwIkQnKGdG3NfSKSnjt/RxK52iWd3xL46DPw7DM7fPzJD3Lu1E5v49oYD/jJp59EifDS+HVevwJPX4WjkI5zAGyNQZXg2jSuYQaqSDqmRZM2bnkOfpG+Y20L5QC2tmDvVMHHnzjLhy88wbDHeUNrzc44v1WpN7OKzY2Cxo+oFhV1jGxmGeUwJ7Yth7XHiKCNwbmAKOHUuCAzhp3NRfKmihoXPWWWsT0sKTJNllmyrF+h0x3kAjBakdnba0/UKvULZGnKnfTFxbEIVursK2RWbrW0iZF+7YpOEHdXKD7ukLcTlInI14BnY4yh+1sDL8UY37I34rvB5z73ufinf/qnJ/2y98WytU7yzUqOulEE6UK1otQdzSwfV3jvWVRNatcQPNE7aueY154QAnmmyLUlEnExplY40SNKp55YCoy2WKPJjcYYjfOe1kWUVpRGU5YZus8ablIofDqbs3/tJm/cmLCoa3IjjMoSozU+JEFodAHRQnCeNni8F2wmbJZDNgYFiOdgUvPda1O+/MMpX359zhuHd4qsf+qD2zx/cY9f+cnzbA9XW+/fNA0Hkzlv3pxw9fCQunEYpRiUljLLU8PZ4KnalqryuOgwoshzS3COuXfEVpAsMlQWW+RIgKwwjPMBW+OC8aDA9qiVAbpm7Z6qaZnXNVXtWNQVR/M5s9ojMVJkikFZkpuMItdYlVpjLVpHXbcEQvKAskKmLZlRWGPIs5xBZhiUWepl2iPquubqwRGvXznkyuSI6B2ZsRRZ0sZIEFocMQjWGLQEgsB8UnG1muJqT1SwkWVkecbAFhS5ZVBmbAzHbA5yNoYFWY+9ESFF76fziutHM64dTagahxawRiAqau8IbSAq0AJKJa1qZjUawUnEt2CsYCI0eKpFQFnF7mjAma0x42HZ+/mCdC3WdcusaqidR4sk0qEUSfInWC0QA/PGsWg8EgN5pslshtUarVLFXtV4Gpdid5m1lNZQFLb3+fBhcHxNe9Da9TCPWeOdQUS+EmP83Ns+7iHJ1l+NMd7o/t4B/uBHnWyt8eOPm7OG3/n6JV54aZ+v/PDOpNaHTg/5/MU9fvXZPT6w01+0ZI011lhjjR8fPCzZephtx/8CvCQi/55Ehv8K8CObQlzjxxtV6/k333qTL760zx9+506h++lRzn/27BM8f3GPTz7x7oXua6yxxhprrPFO8JZkq/PT+mPgL5N0WwL8jzHGk9L1rrHGe4YPkf/3e9d54eV9/uU3LjGrb8t5B5nmlz91jucu7vGzHz7VmzHrGmusscYaayzxlmQrxhhF5IUY408B/7ynY1pjjbdFjJFvvnHEl17e54WX3+Dq5LYDt1HCX/noaZ6/uMff+PjZ1AFgjTXWWGONNR4RHiaN+J9E5C/FGP+/lR/NGmu8DV67MeeffzUZjn73yvSO+y4+ucXnL+7xK59+gp0VC93XWGONNdZY42HxMGTrF4F/ICI/ILXg62yqTl4gv8Ya98PBvOF3v36JL764z5/eJXR/6tSAz1+8wK8++wQf3O3Pg2mNNdZYY401HhYPQ7Z+eeVHscYad6FqPf/2W1d44eV9/v13rtzRpuXUKOPvfeYJnnt2j09f2FwL3ddYY4011nis8UCyJSIF8A9Irbu+DvxGjLG/bqlrvO/gQ+TLr1zniy/t8y+/cZlpfftyKzPNL30yCd1/7sO7mMe0s/saa6yxxhpr3I23imz9E5JR8h+RolufAP5hHwe1xvsHMUa+dWnCCy/v88JL+1w5JnTXIvzCR0/x/MU9/uYnzjLI+jdIXGONNdZYY433irdavT4RY/xJABH5DeBP+jmkNd4P2D9Y8KWX9/nii/v8xV1C92c/sJUc3T99nlOj/tp9rLHGGmusscYq8FZka9lflBijW+ti1nivOJy3/ItvJKH7n/zgxh33fXD3ttD9qVNrofsaa6yxxho/PngrsvUZETnq/i9A2f29rEbcWPnRrfEjj6r1/MF3rvDFl/b5d9++QntM6L47TEL35y+uhe5rrLHGGmv8+OKBZCvGuHaCXONdIYTIl79/gy+9vM/vfv0Sk+qY0N0q/nYndP/5Z06the5rrLHGGmv82GOtOF7jxPDty0e88NIbfOnlfS4dVrdu1yL83DO7fP6zF/ibnzjLMF9fdmusscYaa7x/sF711nhPuHS44EsvJ0f371ye3HHfpy9s8vzFPf7up5/g9HgtdF9jjTXWWOP9iTXZWuMd43DR8nvfuMQXXtznT75/g3jsvid3Bjx/cY/nLu7x9FrovsYaa6yxxhrvL7IVYySESCSp/JVKguy7b7ufUPv4c4kdvRB5y+f0Ae89i6phumionUMrIdOKGCONj8QImRGMUgSECBgRssygVXpcRIgx0LYtVRsIMZmIjsqcLLOICLXz/MF3rvLFF5PQvfHh1jHsDDP+7qfP89zFPS5+YOtEPovl5x1iJPiQfodIiIEQ0tkyWpFZjdYK7wOtC/gYIERECXF5voAYAgEQUWRakecGrR+NLDHGiHOepvX4GFGA0QptNAI451jUjjZEjILcaEQpnAsEIoQICgSF0QprFEqpR3o9xhhpW0fdepwPxOghCihBYkREEKUw6vY5i5F7zm/sHqtUGguSxqKVoLV6pN+zunY0PhCCR4mgtUYJKJWOKx0veJeuNSWCNerW9Vk3jjYEJII1CmvTd/BRzh9r3Iv7rRM/qufnQWP5cRrjjwreN2QrhEDTBnzwVHXDrGlpao+xkCtDkMiiaqmcQwtkxmB1WsR8DLQuYrTCaCFGhTGa3AgxClEUpVEUhe11Affec/VgyuvXDrhyY8a12U1uHkw4mHmaCvIcRkOwSlOWOblSHC4q5pUjBrAGMqswmSEsGm7WkFuwWijyjO3NbbyU/On+gt//9jWOjgndC6P4W588x/MX9/j5j5zCnqDQPcbIbF5xYzLljRtTrh3cpPGB0mYMsozRoERpYTJdcHM2o3E1EcVQa5woINBGQQdPEyISA0d1S6FhPBgzKAr2Tm3w9NkdyrI8seN+GIQQODiaculwxmTa4GKNqyNBArkxKAXzugUvOBWpqhoXAmVmyJSh8g7vA5mxlAZcjMzqFqXh9HiLc7sjdscjRsOyt8kzxsiiarg+WXA0nbF/7YArh0eI0uwOMrTNMGLICoNynhoHTmEz0AhV65i3LW3jES2MspzMGkQgyws2ipzNYc7maECR214XBeccR9M5+zdnHE1nNK1n3jQ03lEog7aKYZ6zOSjRGhatp20DIQZcVTH3nuAhzyCzBVmWISiGpWV7NGBzkGOMwZp+ieTdiy1EvI9pQxJjWnyVuu9C/LhuPB90bCFGXOtxMRB8GpvuNtrOB3yIaK3IjcYYResizqeNEBGUgNUK0TptjJabm0c0rrf7nJeP9SHQtgGlBa0UIhA8aBVxPm1sEIEYkSC9X4MPgx8nUvi+IFsxRpo20LQN1yYV16c1xIDzgaZyBDxN2zJ3nqZuiQS0NoAjxrTzzEST5xlKFIPCoBGmTUuZW7aHJYPCMHYlW6O8N8I1nS145fJNvv2DH/K1H0759qvw7WP3K2AILPBY5gyASDJQmwAW2CFQ0zABAnAqfWIEambqMlU49noCP/fMKT7/2T3+1ifOrUzovlhUvHLliEtXb/DazSMuXbvOvI2ECo5m4AMYDSpC7cC14AQkgrEwKqGtoPbQhjRgF8DksHd+wlOnN7gymTKvHZ9++jxZlq1kHHcjxsj1G4e8+MMr3Dw44MqsZT49ZNKCbsAbqOs0qWyNwRjNtalndgjOJfLctJAVkCnwpMkzRCgtnD93yNM3t9g7e5pP7u0wHA56GZdznutHM169esgrV67z2tXrzCtH3UK9SNfcTpk+/7oFrWA4MugQmYeIIuA9LOo0PjygYHsMHzh7hr2tTa7PB1zwgXPbY7LM9jQux5WDOT+4fJUfXL3B69ducOmao3GgI9gSTg8gWGGghDpAJkI5GDKfTtmfBMSlc6cEtnbgwvYYL5aBVpze3ODps9uc3dlESYYx/cwbMcau12iKfLfOMa89uVUYrWlahwuR3Gq0pIii0YL3ER8j3gd0R8acC4iAVoIP6fVymzajjyK66nwkBE/VtEzmLT56JEIksmgjBI9HkWtQWmOURimFUoF57RLB14ogGiWCD4HZokFrYaPM0UpofaTM6Y1wLce1/DidD/g2ooh3RH6VEkKIyV4nBhoXiICKiqg9MQpKQdsEMAPsbAAAIABJREFUUOpWpDiSWqUpH3q7Bh8Gx8e9jMY5HzGaH8no3PuCbCWW7zlYOBZNixLF4azicLZg0cy5dniEVpoiy/FK0VYNdWioW49RQmktxuRYmZJnBn8AUSJjWxC8AlpmTaD1kUzDeNTPInd9MueV1/b5w29OefkGHN49bhKpAnDA4q77W+DNu267evcLAKcHwt/71Fn+/i98lA+cHp/Isb8VbkznHBxNee3mdb6zf8jlq/DqTXiDtHD7ux5fkohlRRqnupkOPet+Zt3zaGBzAp+5fMRHnjgiOs+F3THnTu2sfEwAdd3wzf2rvPL6Jb57Zcabl+CHc7gEjIBNYM4trsFpPDlwBbgBaGBM+tLeBIrudTeAMznM3JyqmoNSnBoZPtQT2aobx9WjOVePZly+dp0rB46jKRwcwbROx3tJ0jlwMU2MWebQIRFi6WbLKsLYQttCpmFyCqb1FQ7nFU/ubGGiY3NQ9Ea2ZvOaKwdHfO/ydV65cp0rB4H9V9P3LJKutSEwVJFMItGA1YA+4vokXXe6e5wDitfgnJ1w+hwUQzg/mTFvGyKavVPj3ha6lIZfEpPArHY454kYVOtYtBGrBSWCWE1dt8QQyTJDDJHGeUJMpDnGtPgTA2iFdxFjYFzkvUchQ0hEa7JwzKsWD0xnDTemFUQPypArRRAhhpaIYndcUBRl2o0JhNAyR5FnGRJhVje4EMmsYtE4MmvQ4mlbyPN+yFYIx4lW7IIHntZ7tEoygqVeQglEkSRRCJHM6luZHEXEGgMEiiwjzyyqS9mrjiw/ToRgOe7lNZR+J4KlFPclYlolwv84ErDH6bNdGSJpQnCtY1Z5jqYNN2Yzbs5nTBZzbs4qchG09Sgih4uKagFZDmVuUgTLeg4XC8pMMcwK6qaiKErGVc3uouDM9piFEY6qltEw9nKCrx9N+fprB3zvPkTrpHAO+JndyGfOaDbKfi6Xw1nD9aMZr1w55NXX4NICXn2Lx8/v+nsZjGu6nzteG/jqUVrkdXad/WsTzu5u93K+juYVP3jjMt98bcZ3X0/Hcqm776D7OY5r93mN2X3+f50uInYzpatOTRZcvjblQ0+c7PE/CI33HM0XXJ1MePWq4/JVOKzS2CakSc92i8Eysjpu0u+lQYiK6bwN2u58ejj3Jjw9BdFHDLIMbQwXqprNcT8kclI1XL05582DA/avBm5chX3gCKiPPzCt04x8IlUt6ff98P0WPv4a7G6DqxYobrA5GLA5sAwHxQOedbKILDegKUpVu4ASTdU4GtciogFF03qs0zRNizWKPLcsmpbaR4SIEhCEm9MFlQ+cGpdk1tK0jqPYJv1oT8R4Oa6m9dR1zaRucQ6OZg1V7ZhVNaOiIGYZSoT5oiVEmFctO1uRYadNbdqIUgGlI3XruHFUpc1BbohB2NIKRNH6QF/11ZFEKJzz6Xy1AecdLqSInW89EgOtT1FFawyIEGLgaFbTes+k9pQ2Hbf3KTiwUQRya9Cd7vPxoCS3sRz3cRyPaB0nYtDpKpt4SyeZ1v3bkbBHjfcF2RKgblqOqpYbh4e8eTjl8uEh89YxrzyugYWKhHZBE2A2g/kcygyMdeQ5mHzBfApZGcjMHFdDNpxxYQsybSnmHqRhs7SEENF69Sd3XldcvQqXV/gel4HXX4OrH7zZ7YhXDx9aXr9+laMjePNtiNa7wSHw6gx2LsGN2aS383UwXfDqtQmvvp4iVVdO8LXfIF2z5gqc2z5k3kMEcgnvPJNZxetv3uC7r6Wo2zVuE47IvaR3ctffS4J8nDhfJqWN6+/BYnGN4TOBqt098eN/EBZ1zdXJlIOjlqMJ7Lu7Ir/HELl3TA/Ct4DzN2EyhdzMOb89YXpmk7Mnc9hvC6GLRpE0ZikqkAobqjpQ5IbpoiXLDSpEmgCxjUwX7a0UI0DtUqFA7T2+8dyYVkisKQsFUVM1qleyRYxMFw035y3TRc2i9hzOZ7QBXOsQ7VKBkFIsnCPThsYFZrOGqvZY8aBS8cKbN5MO9MZkikExZoBWkaw2lFkkhv7SbUIiElXjCTHiY6BqA6pLA1btUseZCHLjU9FNXbfMWkfrHETNovGE0KAFclE0IZCJUDWeGAODvMdz9RDoWtXcoxdM8a3bBOpW+jiCKAGRjmRJ0qn1NL+/Hd4XZAsik6rBtw2LJrB/eMSicTTOMzkEDEwPU3SgasHHlHKLddqllsCmTSFaM01FVkalFMK1bMZgWDKrNJnOUWpIP5QEjLJJr7Ti9/k+cDCbY3q6YHOjmbWexSyRiFXgBnA0Bd+43s7XrKo4mKbIyI23ffQ7xwFwdQY3Z47S9vfVdt7ResfhLJGqK9wmT+8Vc+ByA7wKH3piQVO1b/eUE0MIgUVbcXOarpVLb/+Uh8YlILYweAPObF7lU0/2RbVSaiXESJNCI1itmbQt3nmiBI6mc8SolDrzgbpucTqlbhbOs6gctfPkCrwIs2lNVJqCQJFbmhZ8CFjtGPcU5Yd0vuZty6KuqWrPtKpZtJ75YkGMniCCdw4komJg5j3BB0DItEYrj8kyWueo20D0Sf/kFYwiOKeY1zVaMkZFf8REBOo23Mq3Bx9pQyATmCw81qSq8rrxaC1En/RcCxfwPjCvI9ZErNUQOlKiNb4NSCloBcGnIoHHCUp16VAfiJKqmnVXgX68cnkZ5YqkCuDj6UbdVeY/DnhfkC3nAsTIwkXm1ZR2UfHmmxA60hQDNDUsmrQ7rUnamJa0eGjgZpvI1VKDMR6Cb+DoECbllM08TwJ6ZW5X6KwYo4E9uVXtLXCDJDg3PQlCy7xg08D+jXsjIieFGsBBIPYWPo8xzZdHrI4gt0BoYHPYT0oKoPGBzGRonYouNCd7WV4BzgWoqoZFqN/28ScFYwwSPE2bonUnjUAqIHjj0OFCfyRSRLBKqGPa8TsXyZRQScS5QBM8UjsOAzjvyTQ0XhFwqCi0IdC2Hp1lEDytOKTxhCwtJyGC9h5E9xpVcD5SGI0HZnWNiwGCY9E2xNgSomfSacy2RgVaDMOyQEkkkMhVTujyrJ5Z26JFoXC0wWGDEILBGMH2KCSPnT4Onyx4fPCpirdxBJK2rm4dIAy1QWuhcZG6aTAqZWhq5ynEUPmAlgjekw9sslQRjdAfKb53fA+2p3BdL93jf1uTKkpvFXnceiGQbolaPn4ZCXscsDKyJSIfAP5vkuwnAL8eY/zHq3q/t0LjPC5AXTVUwWCKAp1VzCcQAmQ26SchEaptBdbCrE47a0MSI1uVqtuEJGAOqWqWEBwEhyZgewxXDq1lMOYuAcnJQy//6enLaBTUwd+jxTppaA1G61t+a6uGVlDmiZCs7D1IAvu+iDGk71CuNEpBThLsXz/B1xcSORVtqOf97VIzrZg1LTSr0UReAXYj6JAqbPuENpoipAhUCB6RFOFJ1jbCtHY0TU05tPiYhMe+bVGZpbSW3OgkqNaGorE0mRC8J3iPsQqVWYw2vUWNIXEkrTW5VgwyS7uocEExyi2eJO8Y5TneO7RocpsxzgdorYl4lABi8NFhsoLcC4ImsxoCt7RNmdK9RoFCjMQoiILCao4aRyRFraxVCIk8B5HOM1EYZAZixtGsIURwIdI0LcEH2uDIc4s1yScuRE7Uuued4O7K2BAjuGRJ5FxAa0GpjthqRQgB5wJZZjC6K/boghuZVTgfu2KPZGEiRqcCgscAq4xsOeB/iDG+KCJj4Csi8vsxxj9b4XveF94HINCEQKmFgcmIoUrC1pBKs42BPEAdoSjTBehIJMsA4zItKpM6RVsGdaqY0gVMK0fTOrQ1iKh0wfQAbTPOn+b+SuoTRABy1VvADh8Di2lg1Xt90TDI+quYEoEsS4R+VViQNg7Tqr8IkDXCLNS3NiGzt3vCu0AGZKJuK+17gEikni24Pus2HCvAHEDAtXfX2K4WyXBV8E2y7ggx0LQepRLx02IIVtAoau8JwWOMZqA1w1wzrxqqGPA+MhgYSp8W+yCRIuusFYReowpGCSEkwb/VmjLLqduGGoUBlDWcHo1xoWXhAoSW1je0DrSxDAuDSMQ4oSFA7BZtgSLLMCqRkiLr19YihohIpHUp6jMoM6bzGmsj4zwjKkWrMkS6KJBKVhzWGIxOlaQqU7TOo1RkWFjGmYIotM5jlEopxkeAZWWsD+lzVkrd8sSMxHtslJb3Q4peJTJ2O8oF3DLq1j1toh8WKyNbMcZLdDKHGONERL4F7AG9k620C+l8fXzg2tERIpAv/SwjkCXNVgDms/Q7kibZ5FWSytinpBSjLSAqmFVQGGijT7uLGPAurDZ8ATQu8B9fucl/WKU6voMATvUW2KJ1wiSw0mqfHLoQpdwjwlwVmsbjw+pSo0togYNZj2RLK1yTvLJm3K4wPClsAoMNKMqcgelPK1M1noWHiVtd8LglzTt1j+lRSN9l5yF4jweaNqCN4B1UbYt3yWuwaiE3Qu0iJqRN66JJhEaLI6Jo2ohoYZhZyszedvoX1ducASTCECPOOWa1o3VJUF7VjsxadssSpQ3RB6wCFzWTqmFgNTYaFlWNNprWRawSjDI00dM6GBQBiQZrdBJh9whRQnAgBEIUvEvarMxYRCmsgiCBSeUwWigyS4gRJYoys8zqwLiwiEpExCpFZhTGpG4a9hGYtC4R6eQVxyoLbxEqn8yejx9bCIG7j1REMBraNnm+WaPvccp/3wjkReQp4CLw5fvc92vArwE8+eSTK3n/zGhyG1mEmuhaNkYFR/OKALSLZBJZ16BzGARQGqoKRgoOQ4pEmAFQpYk/B+o5jM6lFCQFlDZnMq+ZVI5RuZrdd4yRF1+9yRde3Od3v36Jg3k/Oo8AZDHthvuAjy2Lhy3teg+IHhTSX/Vo2xAFdoDXV/QehlTk0fpVl00cQ1Qo8SwWKbJVcq+n23tBA5gApVaMB/05/s/qBkKKbq/q6hiQ5h/f9rvYLUXFjogGmiCoGImiiR7mLrJpFTEKjU/VfK0ojmY1gQBB8LGLkgwMbdsQgqNtBSWK0SAnz0xv0XC47cM0LCx1G6jbhtxkVLbFu5Y2OGZVhVEKQaNVwIhCROG8p46evMtQtE4Qq9hUioWPhADbGzmZ0VRNILOhN4IipPSaUsnOQGmFChFjBO89NxcekUiZpSiQ8xHnW5RocqsxRtCikiO+CEqSpMHqZD77KEVNx8e2RIwxrTVG4TxAuEXAQoAsu/dzFxGUVugHWEU8Dlg52RKREfDPgP8uxnh09/0xxl8Hfh3gc5/73Eo+FWs1RhQGRVQK39QEAVeBV8mB/HCaTvxgA6wH8cmNXOZgs2RWmOlUqai7fEkMkA9htyjI84J5EwkuhW1PEt+9MuWFl/b50sv7vHbz9jImwN4Ibk5Xk7o5jtrT25dSQoocrjKxEgBtoO1clvtACILo1dY0CMkUtOhRpxAkMq2SNiTjZIkWJHJ65jxYk5H1OC7XaT0LDX5FF6MiLXxZ1m+tUupfGWnbpD0dF5ZF7fF1g7UaXTt8jOQGqkWdiiCswmqDcy3z4JNOTyC3mro1ZFooiyxFGYJQKulVs+V9wEdhkOfMas98oYgExlnOVNL9RqVKvQbQUZJWKSpaAgOtqdrUVaQsMzIRFj4Jy43VxCAgSZuYdEP9XYtKJPV7jXTi/Jgc70mGqzFEoijUcnYJaX4LIZ2zZYo/huQen5zSkkO+IBj9aATySkkqVAqJUCVROygFShRGp896GdHKsttRuHtaTsV4jzfX+0IgDyAilkS0/mmM8QurfK+3glIKazRlbtE4pi4SGogajO/8iTQYgSxAE1N0K7NQ7EJdJUJWZJ0OYdkaxkAhsLlRkimNUQFPYtjvFVcmFb/91Ut84cXX+eYbd3LUj58b8/nPXuCnLwz4va98hd//Cnz3Pb/jW8O7tPjkPfjmaGs5Nzr5Rfs4HIlAN77t7ctY5oZQraaybYlrwCctDPL+qhFd2zJvPDa712D2JHAa2MhhUGY91o6mRc0oUmRnRe/xJjCdwKjos7otiZJ9CGgN05lPdhCdO2sERgODMRqrDY11bBR5IroKnFhspx3Ks4zJosLYpAHypGKJECNN6yjzflphQSKQSoQ2RLQorM2gbZn4lpE2IJ5Z0+KNobAGYzNCSAv6vAUyS2wbfHQsKofPoFp4zFDTtB4XAjEEREwfReC3IUJmFVXjaX3AdNb9Vid9cOtTI/vMaJqmIZCapGulEQLOe4gBEUUUKBSdfUJK/ec2NUp/FK16pBtb03aESlJLIZBbqcD7kdr7tfIJXU5SqePViDwWKURYbTWiAL8BfCvG+L+v6n0eFs57ghfKfMiFbcVWueD7+wvmi2ThoGxKI7qY+rM1VUoljscpmjU9AJsnh24dQCzEBvI8YyCGRQxs25xM3RvKfFhMa8e/+sZlvvjSPv/xe9c4HiB7YqvguWf3eO7iHh89mwwr96/ewLikI1s1UmqqnylGgqfWSeC3Ku1/BFRB1/qhny/jOM/werURuxwYF2Btf7vu1kcyKzSTkz9fY1Jked7AQGlCj2RrUGZsDjR6pWcsNYQ30t9CF0JESWTaegQBCRx2VWuZTuLqKBprIoJHBVAmuY+HkEhUcAGlNIWNtMFjoyLPDblORqbee+o2MCz7bdeTGw1dL8TCGlrf4pqAFJoswrgosUrRxEhb1Sy8Z9G2jIuCMG1ZhBYdhKLQVFVAtKZ2jiw3VK3DGo3Vkt6nL8QkIM9s6tfoQtLNDQqT+vu6Fuc0RiVfMJGImESiWp/Sikql1O6yl2Xrkm9XDJGIRlrPqKS3vr7HoZQiz+SOKNXSjDR226u7NVhLU95lVagsSdrS9LT7rfX7o13PzwH/JfB1EXm5u+1/jjH+ixW+5wPhQyAqhZY0dc5rT56DL0EMVA00TYpY1VVXPi9p8mlDSv0El5rpLhzEOWRjWEwbZsOWXZXKcMvCvqPFu/WBP/qLq3zhxX3+zZ+9SeVuE5qNwvArnz7P8xcv8LkPbt/zut6lir2727ysAtViWdW5ekRJaVwtyUxwVWSy1GBsf5VFWW4ZmNXWTihSCvaW4UwP8CGSaU0lSYN0ktfjgLQJUgrmPiCxv5jCKCvQmUbM6kJbZ4BT21D1qCsJMRKiYESIQtdUWtA+LU5ZFHz0KeqRaWwOwTe0QZAI2hi8r5nM54TQUjeO1nrOpHI9nPMISwfv/hY6pQRtDMNcM5lGWtc5oxcGIgQR8B4HRNcya9tkNwCI99xwNSaCspa6cTTeMcozagcDnxb/qnHknbi8TyyjNAi0zoOE1L4nAmiQyKxqaL1LaUMPi8aRa83mMOuahytQyRBVa0XdNUgPUVAE5pVnWMojEcsvKwuXY31Q38Nl1eKy8uJ4O55lU+3HzZx1iVVWI/4xj1R6dyeU0gwKy9F8QalT76TtsUHEoeekvlJ5iuAMTCIXm2OwA8jrZVlqaiA73Eh9S63A2e0hm8MhLmpCgFFukLe5WJPQ/YAXXtrnd772BjePCd0zrfjrHz/D8xf3+KsfO/OWGpXWO2aLJOA96QqwuxGkv2pE74XRoGBYVDQryiVuADKAge0vzRGDUBaKIYEhq9PZDRWolZkV3AfRM3eBzRI2bqbU2EnFgjYAbdPrheB6i0JCIhpaaTZGUByc/HfsPLBhocgtrumPRMaQuoGXhWVROzJj2RkomuiRoBgVsKhbFs4zzAzjwnD5MFBYlSwkY8BkFmlbWhcpc0PVeGZtoCyTSL31kbJnOwGjFVo8VmuGZUakRekc5oLJLJq0gGs0lVYUUSjzgta1TJ2jEE1eWkznaSVKkdmMUZkzKC11GynzSKb7JSRLHVIIAR8SqSyMZdE6tBJGhaZqWhY+UGpFVFBYC13aMFVXpjSr9yFFKFVEUBitu1SbTgUTPWvR7ocHNaB27naLImGpz1pmJ24HA45Hwh4nvC8c5CGJOIe5ZXMwoI3COa05mk0JymOzyGAzucgPO1Ixm6eU4aiEzIAfJaNT72C8kTy3htYwHm6wVQ4YDzJ2xyU+ygMZ5itXk9D9iy/v89qNO4XuP/OhHT7/2Qv80qfOsfGQrSACgjLwNPDSe/6E3hqbG3Qd43uACpR5xuapiqdegz9fwVt8EDg9SKmivqC0YnMw5PR4AhN4hZO3gbggEDOhyPtb6LQ2FNqSD+HUBhwd3dtmKU2X7wybwO4INjZhs9RYk8E9hd+rg4hie2OD7Z0FHzlInjUnRSLHwBmVGlIrayiL/qZiUYL4NEtlVhNiYOYDA5OlNJk1KFWxkbq7IAg7Q482hum8QRmh8AEps86vyzD0DquEpnUYZciN7l0DpLXCaEWRG4ZtRlAK5qCVxjmHsTmEgMkNpoIwsogyjMucMJ2hrCXXGcYKAWHgA0obstxS2AwRT5lZTF/zYIfYkQ/V2TxoZfEhJP2wElofMVpxZiNLvRIbj7XJdNY5j1YaTyoG8D6lERvnKfOUllQ6RY+MUv1q0R6ABzWgDnBLorP01hJJkVrv0uxizePXgHqJ9w3ZGuQZw7LAh5hKTYNDGHBqmFOFyHyx4PqiQgchKs04d6gMRjbnSNXkhe4uAkWeZYgoTpdDtkcbDLOMsiySViFyx+776qTmd772Bl94cZ+v79/pQ/2xc2Oev7jHc8/ucW7znQuac2vZ2cjY2m0YXF+NOBkSMfnA6SFZT5Nnri2bgyE74yOeOg2HV1O05CRggJ8APrIHF86c/v/be/MYSbL8vu/ze+/FlUfd1ff0HKs9SC65O6MRD4mmSNGiSJPWHoQtEz5oSQB9CLQs2DBEGYZl+B8LPkBbhg1QFEUdBA2bXIqUoGN5iF5RpCgvZw8eqxV3OTtHz/RdV1ZmRsQ7/MeLrK6u6enu4XRmVXW/D9CorsjIyPcyKuJ94/d+v+9jUPYe0ZEfTK/IWRks89TFPfojqF+HKzw6wVUAT12EMyurDMrFJcjnWcb6sOTicoFXNbaO64q+SZwyPUuchZstUzR7lHi7yJ4AK8SlJy5uwupqyTNnN1mqykXOjiJa89zmOq9cv832estzt6JAfjeCqwBWu3/DJbh0Ac4tr7I6nKfV7d3MDE1bG40gC6OpdUySN1phbYwgFkajVYxmKQpaHzizXKG1ZnfcUFtPvzDkmUFJ3jmdB4rMUOZm4QOdiJBlmoGSuJyQglzH9fVGU41RMSLlA3ijGZam860KbC73o+eYFwY9jQ+Ketri8QwyTZ4Jhc5j4v2Cp6lECXjwIVbsiURHea0VWWbIvKdp4+/eB/qlQSmFkmjkbYwgojo/quioPq0tELr3dEsUEVAnYDLqIOfqSFWhOrR95q3lnMe7OC164O8W33Fi/LVmPDFiq8gzNgYFRgW0UvTKjHPWYUNgOp1yKy9Z6U1pnILgcMGTKYUoxfqgjwuOSWvJdIbOFDoIxkSnZLSnUD6uQ5Upxo3jk79zlU+8dIVf/dKt6G/ScW6p5KPPX+Bjz1/i/eeG76pPS1XOs5trvHbrKh/cgs/5R2++eBH40Fn4qmcuLcxluNcruLC2yl7TkOe3qZtAfyd6U72dMJnd/u73ZLYJPK3g6afh2QtLPLe5xtIC1xAcVgUX1gbc2hnSK6dktqV3NYrkktj2G3RLQRGrFh82GrQOfHUJz10see/ZDfrV4vrVLzKG5YCNdU9vWFNlN+ldgzM3oilwbxjd+t20q+wlbt8jns+WOwu+bxCn78ctrK3B+lrG5bUV1voVS0s9igWammYibK4Nee/Zs7j2dUwO+s14XoQoumZXxCoxKj4NcXtNPKeDLFYyz5xTjIbVFcirWHzzno11nt5cZnmB/mFxIWpFngW8Fxrr6We6W34sViQOCk3jBCMBrTT9EiZNtIKwQch0TLquiowQYnKyAjKlKDLTTfUsHiWCGM3yoGTSeCpraZxnWDVMG9+tgyjkfYPDMG1b0IrCGKYuUBpFv8yZ1C2NEfI8o9dFs4yEKFYWPD2qREAFguegYq/IIPgYqdICvcLQujg9fEd0KKrcdNHJ+C/PFM5DkQcaGw1gZxYLIUisKj1mDjvCH64qNEZ1y1rdEWJKKZR667TuSfLXmvHEiC2tFb2qQGvNoFcymTSM25a2dbSDgjVraWvLtA0o8TH07CxOYKXKmTSWrd190BqjFa2PEZheqemXOVWe8dkrIz75L3+PX/jCdabtnWF/WBr+jQ+e52MvXOTrn1l7ZHkng17JV12+yM1RTa62MF+Br9QxgqCI0zD7xEGt6v7NBriCaK3wduLkAnBO4JnL8M1fvcl7L2wuLE+hl2Wsr/Z5j99ko7/EWv8G62+MeN82jPZjZZoDlouYRyMOijI6/29111dc6S32eaUfvZKyPjxzBp47f54L68tsrC4zrObpU383VZnz9Jm1WKW1tcd6f48zqyOub8W8pNUejFoYTUFb2N6D7QkUEitirxHP2xli/2fnbwhsnoP3PZXzwctP8ey5daoFWHTM6FcF5zcGtG3N7YlQryyxWk0YbbbsjKFXwUo/o2laRi1oBytLOd47XrvhsA34FqoerK3GAgnr4cJSj/WVNdZ6fVZXe5wdlAvtV1EYhlWP91/eRLQwKK6zNqiZdCtM6G4EU53/nqhoF9MzkBfQyxRIhvgGMSWVUuy0Dm8dVa/g/NKQc53YqsrFTWfPogLeg+joh1VmKg7WIUQ/Jx8wOkYTUBDQrPTjgNw6R8gzWhcjWUZHwSZALzeIcGwl97OBWitFrxBaBWI9/VwTBtHXQivBOs+0dQxL3U2rBQYKqtwgSlMahaegzPTBEmyCMKjMwhPIlRK8u1tsgMJkxIR+mTmxe1Dxb5EQE+KFuF6gucsl3iNBkRWzQwla5Mg+x8edv8/wlqpCkfCW7d7fOxJ2cmJaETlJ6u/FF18Mn/70p+d2/BACzvmDRMEQouQPPmCdY2odbRvD6VppREJcFV1FEzlrW/YmDoeLy0I4xxevTfmVl7f5pX91665E90wL3/b+M3z8hUvMdhY8AAAgAElEQVR82wc251Yq3LYtV29t8eU3bnFla5e6meDahlHd4rs1hwSoW2g7Q1ZvY7I/CtomXqwQn9JrH18vh3BxbYn3nN/kvZfOcWZ1uLCyYO89O3sTru7sM5q0tLZld2/EXj1Bo8i0Z9K2TNsArsVaRxMgSMBNa/Z8dLw3mWB0RpEXDKuSs8M+1aBPlRcs9XI2hj36vXKh0x3WWrZ3x9zcG7G9X9M2lqIA5RU7TYutG5R22FbYbxvqyT6T1jP2geVcsbY0wPucsa3xTUvtPI1r2Rgu84GnznJ2dYmqrOiVixsUQgjsj6e8sTVie7/BtQ3bkxrbtIBj0rSIzljrFxRKsds21K2j0oZ+qUDlaInl3K0NeDw9I3iTs1KWrC73qTKDNhnDyiy0PN05x2TasL0/5sbOiPG4ide/l87YON5HJq1F8Bid0y8ylvo9emVOCLGQZVzbuP7g1LLfTFEq4+L6gLMrQ4aD3rGU3B/lqEnk4STjo6/FZGrHtLFYH9ASV+owmem8ko4vQXl2n29dQEknVnzA+WhoelDp0zmqS+eqDhzY7WglxOLFgCc+vB6nGJl9/z5EuwZR8pbv+V77CHSpM3fSW7wP3dqJ6tjO0aPkXtWLIbCwalgR+Y0QwosP3O9JElsPYnaRuu6K00qOzANHIfDla3v87Ofe5Gc/98Zdju4AX//MGh9/4SLf9cHzLPcW8xQeQkyErFtL3Vh86ESiCCihbSxT29K0UX1lJroKZ8aQGYXuqlasdXHAc7ZbCd7QKwxVmS98MPDeM502jKYNjfNkSqhy0yWnxpuM6p6grfc0TYsNAe883jtsd3M1SlHmpnNdjtUPRinyLCbwHsfN5vDAFXz0LgoiSIieX6KiaaH3HuvifrqbHghA21rq1tJ25dyFiVVFovWxDQohhBi5GtdMrCM4hzFR7ILvlm1RGC3kRiMIQeKA0Xk0Yv0sUqIOXLLb1h9M1+W5PlZRcvj+MHtylllYYVYB0D28zZJ8teoWyrWOunXxulJCUSxWND6J3E88Pkk8zLh22jnOc/2wYuuJmUZ8GEQEY/Q9v5Sbo5q//7mY6P75I4nu7z074OPPX+QjH77IhZXF5V7MiImhhiwzDBaX7z1XlFL0eiW93oNzjwriVNZp4bCnDPdNttXca3IpzwyLS6d+OESEosgpHrFj+IILv+7L/e4PDyLPo9lnYnHcdZ09wbybv9vTwmk414/z9/+uGTeWn/+da/z0b7zOPzuS6H52qeAjH77Ix56/yFedXzrGViYSiUQikTjJJLF1BOs8v/Klm/zMZ67wyd++xqS9U+g9KAzf9cFzfPyFS3zDs48u0T2RSCQSicTjSxJbxPnez7++wydeio7ut/bvGAwYJXzr+zf53hcu8W0fOEO54LLfRCKRSCQSp5snWmy9cmufn3npCn/3s1f4yq27LUFffHqVj71wke/+2vOs9BZXlp1IJBKJROLx4okTW7dGdxzdP/f63Ynu79nsR0f35y9yafUxyTRPJBKJRCJxrDwRYmtyyNH9V75086AEFmBzWPCRD13gYy9c5KvPLz1W5bCJRCKRSCSOn8dWbFnn+dUv3+ITL13hk79zlXFzd6L7d3zNWb73hUt843Pr6JTonkgkEolEYk48VmIrhMBvXomJ7n/vc29NdP+j79vk4y9c4tu/KiW6JxKJRCKRWAyPhdh69daYn/nMFX7mM6+/JdH9hcsrfPyFi3z3115gtZ8S3ROJRCKRSCyWUyu2bu830dH9M1f47Gvbd7323Eafj71wkY9++CJPraVE90QikUgkEsfHqRJbk8bxC1+4xideep1P/e7die4bg5w/+aELfPyFS3zNhZTonkgkEolE4mRw4sWW84Ff+/Itfuql1/j5377G/qFE936u+Y6vOcf3vnCJb3pPSnRPJBKJRCJx8jiRYiuEwG+/scsnXnqdn/vcG9wc3Ul010r4lvdu8PEXLvLHv/pcSnRPJBKJRCJxojlRYquxnr/6S7/Lz7x0hd+7uX/Xax9+aoWPP3+R7/nQBdZSonsikUgkEolTwokSW1+8tsf//Ml/dfD7M+s9Pvb8RT72/CUur6dE90QikUgkEqePEyW2ANb7Of/mhy7wvS9c4oMXH12iewgB7wMBEEB1+V3OeZwPOOdwzhNEUIAxCiWKQMBah3UBBIwIJtMIgihBiaCUHFtCfggB5zxNa6kbS+sdbdtifUBQ5JmQa4MohbWW2lrqxtHaFk/AqIzcgLOWvdpRW0eVKZaqkl5VUeYZZW7IMrPwPjrnqGtL4zzeO7xztCHgbEDEE0Kgbh2TpqFuWlCKymQMehlFluN9wAaPtwEXLM6DVoZBlbHUKymKfOF98t7TNJZpa5nUNdO6pfUeQkCLgAKCIjMa8ZZ92zKZCqIaMgQxBUoCg6KgKjO890waR+MchTYs9Qv6VYExi720m6ZhezRma1QzqaeE4BA02iiyrl/eKZSJN50ggnUBozW9wpAbTWs9o7pmf1zTeIsKil6Vs9LrsTyoqMocrRefNvCgewfE9AatFSKCc46mcThAhYDWAqIAmKWVzmp7Dr/vtHGv7+U09iORWAQnSmw9u97n1//St2O0eqTHDSFgXWB2H2itY9q01K3FB8GoQG1BBEQUCk/tAjmeiYMQPLnRiCh8CGRa0cs1KI0W0EpR5BqlHm27H6Zf07plbzzl1l7NzmiXa1t7bE+nBOepjMIrjRZFL4OJA9sGPBbrhczkLGWKG+N9tsdjjMqoMkXjYKlfcHl1g4ubQwZVxXIfinxxgqttW65v73N9a4crN7d5/eYNtvdqhlXGcNCHAI2AOM+ktbTWU2QGrRRBLIUUGO2ZtI66aZm0Lf3CsDJYZrnqsbk24P3n11leGiysT957RuOGN2/c4otXr/OVN7eYNhOqsqAymnEbCMGiRWGbmps19DT0Cs3VbcdkCit90AK1g34GJgOloCpK+v0+T68v8Qcunef82nBhgqtpGl69OeLW9j47ozGvbt1iazxGXItr4zUkATZ6mnHrGU0DysDGUk4uObv7I3YmxEHbQdaLoqRXlSxXPc6urHB5Y4lz68usDauFCq7ZvQOisLDOY63De48NAe98FE4CVW4oM82kDQiBgDBtWlrrGVYGUZqmtSgRyiJDieB8wATIzPEIrt+vYDp8TxWRg9+N5rEQXKdNSD5se09bvx4nTpTYGpTmkQstAO/vCK26sYwmU67tjGnqlrzIaZsGnRmqLMcHy7RxTGuLw9HLM4JoRIR+nmG0sLc/ZZRrMiU03oFXDCvD+lKPPF9cPlnbWrb3a27u7PHKm7f54tWrXNlqEAvjMezXMKiAFmoLZQlZBeMdIIeVHkxrmExAacizGtt9T6tDqFuPKDi3riiMkBkdn9LnjPeeG9sjvnTlFi/f2uKN67d545ZlNAJnW4LfJjfgPbQWxnWMFCggM1DlYMoxwYP10NZgLdSuYX3pBpubcGHNMJk8zTd84Bmqqpx7nwCaxnLl2g3+6Zev8PIrt3jjNtzcgl1qSoiREOJN0BPFhwNaHBmw2x1nE6iBPeA2UADPMeXpzSlfWrvFtG7p58+ytrK0kH7tjqds7464vr3Ll27e5vr1Pd64Bi/vQQssd/1qcQyIYtEHKLOGxjdsubjf7MopgELB0nDK+tqUrY0xTT1Fa02VaQb9aiH9gnjvgEBrPa31BAKjuqVpGlqvUXi8KFRw3B5NEd8iylDmOSZTeBcYTVumTU2R54wmNSF4lvoluYn3k15u0CpbyLV1mPsJplnf325Qnt1TZ9vizziQL7ofj5rTJiQftr2nrV+PGydKbM2LmRvXdNpwe3/Kzd19xpOAMZp63LI1GpMrRZaB95q98QRjFK0NZKahl2dordmTKRI8o9aRBYcpKoZFRlEWtHs1tRMurLIwwTWZtuxPGl67ts0X3rzGazcabt6Am+M4SO8D2Q5MiYNz2IGSeNIzYEgc1B2ggUrHDW2A2xXU7hYaoVcU9HPDoL+QbtE0lmvb+7yxtc0bN27wytXAa9dif2537fXdv+bomy2sWghjsMASMCEO5g1waRdMCU4svHKVpzeXefbS+YX0a1w3vPTKFb74u7f4/Gt3xNJb+vAAto78XgNfAHZuwKUtKIrXuXxmbXFia9Swtd/y6tYWL7++x6tvwKs29g3gzaNvmF2QbfyREf/+fLcpAH0P53bi76KmiNpmbWmJtWG1ULEViMJiWrfUzrKzN2V7UtM0ln6ZobWh9RB8ADyjSc2wgv3G451jb9qAiwJkbVCx3wb2JzWj2nF2JUZVnY/3Ij2HB8378XaCyTlPQN52UI6/exBBCAdCbPbaSeH3G8U5bULyYdt72vr1uPFEiC1CoLGeUd0yri2jfYsNMcrVWAce9l3LeH9KlRWMrSXUnl5ecHtvj9taURmD8wGvoEDY9Z6idvj+gCFCkWmCd+yOG9azbCFPCrVzjKZTXr5xk1ferHnjBrzcvHUwPszeof8LUXBZ4kCHi2LMA8MJmOuwXN1kY9BnpZ8hDOfWl8NMW8vO3pjXbtziK1cDX7oGv/cO3n+4/6Mjr+0So0laYNzb5/VbewsTW9t7I7746i1efgNeJ4rAR8ktILOw/Bq8+b5tvoanHvEn3Jva19ze3ebV67tceRW+zN1/Zw+i5UB3HbBPPM7+DrgWKj3mxuoOT59ZflTNfiiC9+zs7XN1b8p0atmftmhgd9qyN2kw2lBpzcRZCDBpJozHU7xS5NrEXFDvQRS19eQ65nuOJ5bdoqXKNXjHsMjIs8Xejn0IhEA34XlHjFjn0fpoJCumVggQZ1UDSis4yL2L+52UIfvdRHECb93npAnJwzyovTPR2Trf5RjfLbhOar8eN54MsQVY6xhNanbHDeO6jX+AITBpGqbtlNujCT44elWFaxtqr6ibGlGaejrhlmsYVH1wLbcE7LRhUA27m9QS3gd6ZUZj/eKeFELgxvYe12+PuL4NVx4gtN7ydu5MTc2Ydj9bYGMMe/twdW+P99jVg8TgeVO3Lbf393jliuWVdyi0HsRNwLSgbsHqMuw3R4f5+XFjd4+tPfiKe/RCC2KE6xpwfgSjyTuNl/3+UV64PR7xld+DzxIjj4+Kq4CMYXgLLp/fj8lfC8J7z6S2XNsdsztqaBvPXlNjVBajVvWYMsvZ8oFMDEEE7yy37ZRCGXpFhXWWaetY7ZeMJ2NGStPPKsrcYG2gUQGVCW7BA14srIlFP0qpAzGiVWzHgUCxjtrGAg4fAkogoDAKbOvJsxiNaxqHKEWm4/uOe1rq3URxBN7ShxDCiRGSR7lfew+LTiWCDwHvuCtKeVL79bjxRIitALTOMW0dTdvSuIa92mKtZdpMubm7R5EZBlXO1v4Yo4RMAvuNR9yU3WZK28RqOKU147rBBU/TOpwElFLkekhd1/TzCh8Ci0jhzbRie7TP7V24tR8H2keFBcYW9sZguhvyom6gtnXsTWom0xitedSM6Pq2C5X2c/iEt/ncUUOVvbOozztlCkzbmLu2KJQSpnXNqH20QmvGGLh1CybjCWaBRSht66itxTlBm4z9yZSpdVjXUmaAKCbWsj+tGeYl2mhaB8pDGzz7TazK1EZoW08bBNc0GKURGzC1kJuCTBZ/G56JDufvDNQheJwDoyUWALhA03pESSwqsi4WAxnBhzjN6KwjiEKA3MRzcxLygB422nOvKUal5KAoYvaeEDixU233a+9h0akUeAdRdMbCmpPcr8eNud25ROTHROS6iPzWvD7jYfHOU7eOTAmt87StY7+esrV7i+39fQqtKfKcIi8wSpBgMUrR2Cn7bUOuNCKwParZGo2RTCMBRnXL9vYWt6ZjxuMx4zZQGNXlb8wfrRXTdsJofCcN5lHSAvUEtIoVlwtDCc7HG//0wXu/YwKxb1UPhr3F+bepXIGPQnaeZAI6LE5EoqCZtHMTkS1RcNWtXehTuPUxMd5kQtO27DVjlBeMEprW0csMucCgLKI1SvDUPooTISABclNQiMZhUSFGvWtrCQG8dbEiWsnClxoLxAcoo+UgAqJE0DpaUTgX8F1fAGoXyLSOFZQhYH0c5D1RnGVGH+RticwKC46PWZ9mzKJ0rqsmjcUOnQAjCsTZ/iJy1/cixD4ed7Tu7bhfe4+KTumie611EMKJ7tfM1sg6H/MIT/l05zxH0B8HvnOOx39oYopBYK9xuNZjTEahhYkHLcJKv6TKC3wwrPcGoAx7+yNMphgOeqgspyxypq4bqKc1e41nMvW44Bnvjhi5mmHmQczCEhec9bQ+UBSxgutR0hKFTggwbi0S5hGzuDeZ0vi2Yb+ezx9oHzjbh6c3c4piMZWIAGu9kq3d+Qjjw6xvwCJlSbAwbt+aH/eoGNNVaYqibhc37Ru8Z9xaXOsY1y2tdYydxbUtRmeUmWHY63FpdZm1pQF5ZhhkGucsY1uzPx3jrMX6gJZAnhsyrcjwCIHWBZyz6E7oLJLDU09aK4xWqM43UDrRJRIHcO88mRK0jvY23t8ZtH3ne3E4xWA2yB8nSknMRwtRRLU2WnRoraKv4iExeFQgnkZ7hMPncVZo4TqRcmBV4gIc2u8kTyDOpj/fThCfRuYmtkIIn+JOQdKxEi8mj7MOCOy3DSEIfZOjxbM1HjOpa3SwaKNYNoYs12gUmSgy7cmyjJW+oanj1JpvoKqgzEuWl5fJQ8a+BSV+/qNphyWwlFUMh/OZvmmBKgOtMqbt4p4sikLjtJD5+eQ2FcCwhCA5eoE5QIMyp55HqO4QfWBpAM4u7kZqsTh3x7phHuRAnquF5thBIHjPqPaUJmOpN0DjqZ1lkAuFMWQScASUszhiFV8vKxnmFVlmcAQkODyeSinKokCUQRtFkRuM0Tgfc2kWyWExAhxMPc1Ek5IYbcuMQpTCGNVFFzxGx9fsTISpu6MnJyEP6HC0xzmPkkNeZp0RtT8iuGYP5ad9kD/cB60VPkDdxLEvvk5npHv8Eci34145dye5vQ/DE5GzpQRaJ5R5xo5MkRDYHU8ZTWp2pi1Y2FxR9IuMcT1FKxU9twhYAe/izaXUij0dn3b6pQYPQTTBe8rCgA/RuX1BUwIKRa+fo308kQUxSfpRYQAnUClF6xdXImxEMzCGYKCdg4q8CeyPF5vXBGCD0Osxn7nRDgFGE8iLxU37eifkJcxzQraqQJTG2gXebEUxKHJ2iwbvLGVQbC4N8ViWez28tziv8F5hJYqpFst+O0arkpWqR25yciOMpw1jHyh0hlVgxGCyaIJKlxO1SKIY6by0OnE0i2bNokFuds1L6MbpgISAljigZ0ooCoMPd5LiT1J+0yxC99aptG6249C+s+/g6CAP3aoPnoPo30mPch3tQ2YU3scpbKPlrkrT4xaRbxdFPG0VoQ/DsYstEfkB4AcALl++PK/PoMw0uXimTc1kOmXSTpjYluChbuC1mzXjuubccoXHsN/56qz2M8AxnjQ4HxgWQtMGtnccKwNYrQqqoiIEwRhzkPuwCIpMUagMF+KJfNSfmgOZhp3WgvcLmxoIAo1zTGuomE90K8vj30U9DzX3NuxPG5aWoLo9nz5BjHBOp7BULHCxdhXITYyqzYumhnrSUOSLE5FKBGM0a/0qrjRhobUN1ita6ygLwyAzWOuZth4jGmVyWucosxwRTeMdrvYoLVhnaYhCC4kPgVprcq3vmtZaFAdJ091gFwfpGBXx3nc5MzFep0UxKDQ+CAgIQtYlyusu0fqoaDspHK3WU0porT9YOumwQHQ+3CVEYiVf9zsnI/n/Qcz+kqJfGl0OlzqIdB3sd8wRyPvac3D3OZvtf3K/9QezWBe9exBC+JEQwoshhBc3Nzfn8yEi9HLNxClypfAI3sWKNNGdSaaH/SncHNXsThpEZ+SiGU0njMZTJq3HuUCmMnpFRpVDphQNEHxNdJhXyJH8hXmSGU2ea9omlsg/6oCJJzqwi7XRm+URH//tqJsG6wO9HnOp6lwCULA7nTJd4LRU41pKEwXkvKiAfgmiFvgc5Tx2jrPnChh52Gs85QKX6okRgRgNGJYFRocoLgA6AaJEHeTBiIJcK3JlGDct47qmbqaM2gbvPbnOACEItHXDpG5w3nXRhsXfiu81ZRbXTXXxuleKLM8wOq7sobvpxNxoilyjdSwcmk1LzfKFTpoQOTplCt2alOqO9YE5lKM22+/wyiPC6ZjKCiHgrGNS22g8S/RTa1pH27o7+VuzIoFwfFOk95sqfNA092nk2MXWIhDi8iBLlabUCqUDrbeIg0kNqojrsbUCN/Y826MaHJisorWOLM9ZqQrKLC7P44NnUBm0NuDi4s1Il69RLG79QOs8bWsJzCc5eZ9YjWh0QOmcRd1Dx9MWJKdfzicPaBcY7cHt7ZbG2oXdbFRQtLoTe3NCgGEfWKAosT7Qy7K52Z0cOMt7aNy8aznvYIymzKIZqXMtTdsSvCWIItea1gnTNlY6h26h9N1Jg4in9Z5J0zBuPd62jK2jzDX9omDatrQ4ytxgJFZI58cgtu41ZdZaR9O6uyrzdBcVQeRgOTXnY6UYLCxF9aG4VwXbLH8reE9dt9SNJXiPUvIWgXhXYv3BMe8M8ich+f/tuJd4ni0zhcT8uxACkzqe38zcMaU9DsH1tlOFnL6K0IdhntYPPwn8GvB+EXldRP7svD7rQSgl1NbHp9A8Q3kYTz07IxjvR++RvRHs78VIigsw9Y66qdHGoBDyTCMmp1eWTKxj0lhujxpubt+mbmpwDSHYhbpA709iFE7n84mWlETPpv16SmXiTWcR1NbRz3Vc23EOx58AXuK6irNy8EWgDYSmc+ufEwFoLJgFDgkOyNV8ijRmrCoYlJrt0ZwrDI6gVFx0PtcZVVEwKEuKTDOuW+qmpa4tTdMymjSMxmMaN4lRcO8wWpF3VXy9PEOrDKVihMg7d3BBaR2F3aI5PNgdLLjd2Tk4H5g2MRJyOB/rqGVC2xmengTul9weQogeaFqRZdGAdtp4vL/7ajw8yHeq665B/iRPZR2IZ6XIjIompj6enzzTiFLREzJTB76JxxmtO2rPwez3TjDPpta1khMZMX2nzE0ZhBC+b17HfqeICMFbRtOWXBnyoqCfw402Jko7ByYGqajruAizdXEqUQVPkRt2Jg2lBiTHiDCWENfnU4peXrHXCtd3plzetGQLElwTaxm3Ddj5nMjZmnVTG4hL8C4GkRA/2b3V4f5RoYFeCdCtTTenzzmMQoGZrygxxEXFF5mgrHzg2l77SIszjhJ8nDYf14uLbM0GIKMMWQ4DldPYwP6kQRvdjcU+DlxNzXYTV0S3IVBqQxs83rWog4WqtwliMKIxeZTDnu6J9xgGksN5MbOBWhEj5jGfS8WpKC+URojuYSeX+7nGO+dRKopnmP30WOvJj+QBzhLr75iFRk5S8v+9mIlM4U4l4uHzNSsMmK0YcLD9mBLP72XGOrvm1L3yuJLYOh0EwCGUZc5qNWRzZYWr29s0DQwGMOxFoVW30cTz/MoKTWuZekcv02iJWXt10FS9HNVYqDJ6eUlVlFRZhg+evXFNVS3Gu8m1cSkNq2GZR+sgv0JMeDZ9GBQ549ov7IkuUwoLDPowmML2Iz5+CagMlvs5eoHTbdoYNqpYtZfzzhegfhhWgfUBKLW4BHkvlkkd+7XGo/d7KYG8FyOeWi9uUPAh4BG0UZSYzp+pQcRhtCHPM4wWnPUEgTXrGVlLW1uCaAqtGbdTggv0vGCdjoaneU4vz0EMRimsD3jnYcHRrcOD3aGhlzLTuFm+jMR1REFQWqE7YTaL8GTm5GSi3K+CzXNHaM2InmFvH2e+X8XmSWQmng+fV7qoXhB94Cjvvb+riOu4onX3+n5VTI67p2A+qSL3YXlixJYSzbAw7NUtVaFZ6w945uyEvZWGqtfDt45C1+RGGAwGrPf7eAHnLN4H1rOM7cmU9czQtCVX3C59k7M+GKK0oVf1KbPAeIEl3CZT9IuCtT6sVrAxibYG75YN4KKGpT6sr0JeRBuMRSUnmsywVBU8d1mz2zhkD157RMfOgWeB5SEMqoqyyBfm3t0rcpbWVri8tk17G77yiI9/AXhqA8oqi1YkC8JZw+WzPW5sjxl1lZaPqtrSAO8BVpegKHKG5eJMaIOPNgeuy1Va6hUQAtZaVA55FgtislJTN5ayKvFTSy8TaucQiQNIYWJkeKksQStEFFppylzwAlrpY4kYHR7sZlOBxigy0Qd5T9KlUMwMT09aRdth7lfBpogWDocFl/f+gXk0syjXaWAmskRiftMsPSJ6ps0S0OPaiLq7PRx3tO7o9zuLqh7d5zRbPsx4YsRWkWly6ymcp98rWXcwahtWvKPMCiatIyMwLDNEBVZ7PTKTYTJoWotRQl4YlNeEntC4htYLRW7oZZo8izdfs8BbT5UVXFxbYXt/zO54m+ZV6Lk7i1Eb4k0mo7NxID79DYnVizlwqYq+TBPi+wKwTnQhLzJYXzYsFX3W+uXCnuhynbGxNOTSxiZ7k6vIlZjQ/jCLbGfcWQ7n6OW5DlwG1s/AxjqsDwacWR4srBJspV9xfmnI5sY2tYV2F648omM/BXzgLFw4IwyrZVb786x5vJuyl/P0xjo3d8dYD/k2vM67XwPSAO8Hzm/CxTNweW2dlaXhu2/wQyJdvpUJjjbEhZurwiBS4Hwsuon5LlCVGuc0q8OcunaEtiU4y8awT5llKCWURU7TxsTsIs8oMo34QJXrY4uWHJ0ym0V6tFYHy/nAHQ+kk7xm4P3aJ6KYNh7wBxEt73nLFOJp5q5IEdHuIc/iuZlFj5QIZX6nCOCkReseR8uHGU+M2KqKjNoGvHc0VigyODOoGLeePCs4v5LhfVyPZ325pF9V8UKV6D7vUSwPSvbqFls79od92tqyUhb0ewUikGnNUm9x0ze9MuPs6oDGbqKUoPUWy7fBTuNA0Ot1C486UD3AQl5E76xLLX8MwtQAAAwxSURBVDQtZBWsBrA11A6Mg6IHRQXDCi6vLfP02WXWBvN0UbqbqjBsDCsm9TCWzBfX4YtwdRqn3oS4hEtOFI4l0M9jv65NorHrCpBraFys1BwCSz0YrsKZVXj2zAqXz66wuTxY2I1m2Ct59vw6b+7soNll0IO1q3H6d5dY5JARK+8Mdy7OJWK/94niud/1fUIUlCXwzAY8c06zPqx438VVlvuLW/Nxs19yvdfjuXMbFGaLKz0Hb8TvvU802x0To64ZMU/JdO2f5ZHMzmPd9XMNOD+AjU3YXM14dmOVi5tLrPQWF9lS3eClRKOV4HQ3QOuiG8yF1jkmraMwOUpBPbVIaSgzwamCTEGRR/NS7xVKHEigVyiMin/rRuuFr414lNlA7RBaF1DCQeXhHcFysqfV7tc+EaHMwVp/ENHKc/WWqcXTzttF4k6KIH4QJ13QvxueGLGVZYalHhQmLppaGhj2chrnsM6BaHJlqMqczWEPpTWTusUFYakyiCgamzNsLN47zq5U7E6neC/0K8Og0PSriv4CB4OyyFgZ9PBAVVacXx5yZXsX27aghEwbMgW705qmbTDaoALUBPomIzeKibOYoLAGKgcuz3HTFlXkXOj3ObO5zDMbK/R6j3r1xfv3a31lgA1dH6uCs/1b3Nwb07Qx4uamcbDOJCa6myw6+otzNIB1UJQK5T0705gLsDzM2ej3We4vcWF9iefOrVJVi+tXnmc8c2YN7yxFdo1bu7s8fbalbj2NixESE2KCuwh4BdbCuIbcwHLZCWINK6XGBcftGi4u9zi/tgGiGfYK3ndhnaLIFtav9ZUhT009mViM0gyqHd57CUoT2JtaECgksL3juTGGUkV7Cutgv4HKRJPZehzP6bKJYl8ZYXUw4OLaGhurQ86tLjHsL1BsqejHp3U0Nz3w+pFA03qc9+RekRuFDxlLVc7tcc1ov8WYjEwJTYjeW4XR1DYw0BBEMASK3LBS5WRGH4vP1lGkM3HV+m5X78OC6qRPq92vfUqpxyqS9Thy0gX9u0FO0lzoiy++GD796U/P7fizagffGb+11lHbaP5mtKbKDXmmCSGWP0uI89+BO+Z3ogRC/Bl8rGbxEo0O887ob5E455hMGkadSWLbtlg/K5sN5Fm86VtvmUwDLRYTovhUqrN+FiH4gA0W24A3noKM5eUeK1VJv1ceS7/Gk5q9SUNtLYr4tN1Yx7RxBAnoEPASsFZQGqrMUBY53jr224b9sQUJFCZeqK0TTKZZ7ZWsDftUVbHwJ9sQAm1r2Rntc21rj63RhNa1VMZQljmZ0iitcM7RumikG8QhQZFlGWUWzSNFZTGfyNXsTR2NE5Z7hnMrQ4aD/sL71TQNu/tTdsY14+mEEOgiNgHvwYsiE0ErT90GJtYheJT31N5jrVAUipWypKxynA1M2xbrA2Wes9IvWOpXZNniRCTcZzmRQ9tnScg+xL/b1lraznJAd1VVnrigM12OVqYVeR6jWqdhCZhEInFvROQ3QggvPnC/J0lsJRKJRCKRSDwqHlZspZhqIpFIJBKJxBxJYiuRSCQSiURijiSxlUgkEolEIjFHkthKJBKJRCKRmCNJbCUSiUQikUjMkSS2EolEIpFIJOZIEluJRCKRSCQScySJrUQikUgkEok5cqJMTUXkBvDKcbfjCWCDuFRd4nSQztfpIp2v00M6V6eLk3i+ng4hbD5opxMlthKLQUQ+/TCOt4mTQTpfp4t0vk4P6VydLk7z+UrTiIlEIpFIJBJzJImtRCKRSCQSiTmSxNaTyY8cdwMS74h0vk4X6XydHtK5Ol2c2vOVcrYSiUQikUgk5kiKbCUSiUQikUjMkSS2EolEIpFIJOZIEltPCCLylIj8ExH5goj8toj8+eNuU+LBiIgWkc+IyN8/7rYk7o+IrIjIT4nIv+yus2867jYl3h4R+QvdvfC3ROQnRaQ87jYl7iAiPyYi10Xktw5tWxORnxeR3+1+rh5nG98JSWw9OVjgvwghfBXwjcCfE5GvPuY2JR7Mnwe+cNyNSDwU/yvwj0IIHwA+RDpvJxYRuQj8Z8CLIYQPAhr4d463VYkj/DjwnUe2/UXgF0MI7wV+sfv9VJDE1hNCCOHNEMJL3f/3iAPBxeNtVeJ+iMgl4LuBHz3utiTuj4gsAd8C/HWAEEITQtg+3lYlHoABKhExQA9445jbkzhECOFTwO0jmz8C/M3u/38T+OhCG/UuSGLrCUREngGeB379eFuSeAA/DPxXgD/uhiQeyHPADeBvdNO+Pyoi/eNuVOLehBCuAP8T8CrwJrATQvjk8bYq8RCcDSG8CTGAAJw55vY8NElsPWGIyAD4aeA/DyHsHnd7EvdGRL4HuB5C+I3jbkvioTDAC8D/GUJ4HtjnFE1xPGl0uT4fAZ4FLgB9Efn3jrdViceZJLaeIEQkIwqtnwghfOK425O4L38E+JMi8hXg/wL+mIj8neNtUuI+vA68HkKYRYt/iii+EieTfx14OYRwI4TQAp8A/vAxtynxYK6JyHmA7uf1Y27PQ5PE1hOCiAgxn+QLIYT/5bjbk7g/IYQfCiFcCiE8Q0zc/aUQQnryPqGEEK4Cr4nI+7tN3w78zjE2KXF/XgW+UUR63b3x20kFDaeBnwO+v/v/9wM/e4xteUeY425AYmH8EeDfB35TRD7bbftLIYR/cIxtSiQeJ34Q+AkRyYHfA/70Mbcn8TaEEH5dRH4KeIlYqf0ZTvFSMI8jIvKTwLcCGyLyOvDfAv8D8H+LyJ8lCuZ/6/ha+M5Iy/UkEolEIpFIzJE0jZhIJBKJRCIxR5LYSiQSiUQikZgjSWwlEolEIpFIzJEkthKJRCKRSCTmSBJbiUQikUgkEnMkia1EIjF3RMSJyGdF5LdE5P8Rkd4JaJMSkf+ta9Nvisj/JyLPHne7EonE40cSW4lEYhFMQggfDiF8EGiA//jwixJZ2P2oW3z4TxGXavm6EMLXAh8D3tXi0d1xE4lE4i6S2EokEovmnwJ/QESeEZEviMj/QTSXfEpEvq+LMv2WiPwVABHRIvLjhyJQf6Hb/ssi8sMi8qvda1/fbe+LyI91karPiMhHuu3/YRdV+3vAJ4HzwJshBA8QQng9hLDV7fudIvKSiHxORH6x27YmIn9XRD4vIv9cRL6u2/6XReRHROSTwN/q2vs/dp//eRH5jxb43SYSiRNIegpLJBILo4v8fBfwj7pN7wf+dAjhPxWRC8BfAf4gsAV8UkQ+CrwGXOyiYojIyqFD9kMIf1hEvgX4MeCDwH9NXN7oz3T7/gsR+YVu/28iRrJui8gl4FdE5F8DfhH4OyGEz4jIJvDXgG8JIbwsImvde/874DMhhI+KyB8D/hbw4e61Pwh8cwhhIiI/AOyEEP6QiBTAPxORT4YQXn5kX2QikThVpMhWIpFYBFW3TNSnicts/PVu+yshhH/e/f8PAb/cLQ5sgZ8AvoW49M1zIvJXReQ7gd1Dx/1JgBDCp4ClTlx9B/AXu8/7ZaAELnf7/3wI4Xb3nteJYu+HAA/8ooh8O/CNwKdm4mi2P/DNwN/utv0SsC4iy91rPxdCmHT//w7gP+g+/9eBdeC9v8/vLZFIPAakyFYikVgEkxDChw9viOv/sn94073eGELYEpEPAX8C+HPAvw38mdnLR3fvjvO9IYQvHvm8bzjyeYQQauAfAv9QRK4BHwV+/h7Hfbv2zfY72o8fDCH843v1J5FIPHmkyFYikTgp/DrwR0VkQ0Q08H3A/ysiG4AKIfw08N8ALxx6z58CEJFvJk7d7QD/GPhB6dSciDx/rw8TkRe6qUu65PyvA14Bfq1rx7Pda7NpxE8B/2637VuBmyGE3aPH7T7/PxGRrNv3fSLS//18IYlE4vEgRbYSicSJIITwpoj8EPBPiNGhfxBC+NkuqvU3DlUr/tCht22JyK8CS9yJdv33wA8Dn+8E11eA77nHR54B/lqXVwXwL4D/PYQw7fKuPtF95nXgjwN/uWvH54Ex8P1v05UfBZ4BXuo+/wYxYpZIJJ5QJIR7RcsTiUTiZCMivwz8lyGETx93WxKJROJ+pGnERCKRSCQSiTmSIluJRCKRSCQScyRFthKJRCKRSCTmSBJbiUQikUgkEnMkia1EIpFIJBKJOZLEViKRSCQSicQcSWIrkUgkEolEYo78/7YwlcINrI3cAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.regplot(data = df_score_rating, x = 'ProsperScore', y = 'ProsperRating (numeric)', x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/100});"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> There is a positive linear relationship between Prosper Score and Prosper Rating, ie a higher score leads to a higher rating. Therefore, we can check the relationship between other features and either one of Prosper Score/Prosper Rating interchangeably for the rest of our exploratory analysis and for the data after 2009.\n",
    "\n",
    "### 4) Homeowner vs Prosper Rating/Credit Grade\n",
    "> Next, we will assess the relationship between being a homeowner and Prosper Rating/CreditGrade. For easiness we will create a variable called rating that picks up either Credit Grade or Prosper Rating. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ListingNumber</th>\n",
       "      <th>ListingCreationDate</th>\n",
       "      <th>CreditGrade</th>\n",
       "      <th>Term</th>\n",
       "      <th>LoanStatus</th>\n",
       "      <th>ClosedDate</th>\n",
       "      <th>ProsperRating (numeric)</th>\n",
       "      <th>ProsperScore</th>\n",
       "      <th>ListingCategory (numeric)</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>EmploymentStatus</th>\n",
       "      <th>IsBorrowerHomeowner</th>\n",
       "      <th>Duration</th>\n",
       "      <th>months</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [ListingNumber, ListingCreationDate, CreditGrade, Term, LoanStatus, ClosedDate, ProsperRating (numeric), ProsperScore, ListingCategory (numeric), Occupation, EmploymentStatus, IsBorrowerHomeowner, Duration, months]\n",
       "Index: []"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check that there aren't listings for which both are N/A\n",
    "df_loans[(df_loans.CreditGrade.isnull())&(df_loans['ProsperRating (numeric)'].isnull())]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ListingNumber</th>\n",
       "      <th>ListingCreationDate</th>\n",
       "      <th>CreditGrade</th>\n",
       "      <th>Term</th>\n",
       "      <th>LoanStatus</th>\n",
       "      <th>ClosedDate</th>\n",
       "      <th>ProsperRating (numeric)</th>\n",
       "      <th>ProsperScore</th>\n",
       "      <th>ListingCategory (numeric)</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>EmploymentStatus</th>\n",
       "      <th>IsBorrowerHomeowner</th>\n",
       "      <th>Duration</th>\n",
       "      <th>months</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [ListingNumber, ListingCreationDate, CreditGrade, Term, LoanStatus, ClosedDate, ProsperRating (numeric), ProsperScore, ListingCategory (numeric), Occupation, EmploymentStatus, IsBorrowerHomeowner, Duration, months]\n",
       "Index: []"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check that there aren't listings for which both are populated\n",
    "df_loans[(df_loans.CreditGrade.notnull())&(df_loans['ProsperRating (numeric)'].notnull())]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create Rating column that gets populated with either credit grade or prosper rating\n",
    "df_loans['Rating'] = ''\n",
    "\n",
    "m1 = df_loans['CreditGrade'].notnull()\n",
    "m2 = df_loans['ProsperRating (numeric)'].notnull()\n",
    "\n",
    "df_loans['Rating'] = df_loans['Rating'].mask(m1, df_loans['CreditGrade']).mask(m2, df_loans['ProsperRating (numeric)'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "#store totals per Rating and Homeowner status\n",
    "df_rating_home = pd.DataFrame(df_loans.groupby(['Rating', 'IsBorrowerHomeowner'])['ListingNumber'].count()).reset_index()\n",
    "\n",
    "#store totals per Rating\n",
    "df_home_aggregate = pd.DataFrame(df_rating_home.groupby('Rating')['ListingNumber'].sum())\n",
    "\n",
    "#merge\n",
    "df_rating_home = df_rating_home.merge(df_home_aggregate, on = 'Rating', how = 'left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmQAAAFACAYAAAASxGABAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3Xuc3fO97/HXJ2OII24lwha5IFqaIXJl2xlJ2aFOTwhbG72gbsXO1taWak97Uju73ap6cVopbd1KlaCl2W16cS1alyTELRFCgyl1CYe6hCQ+54+1ZjrGJFmJ+c1vMvN6Ph7zsH63td5r1si85/u7RWYiSZKk8vQqO4AkSVJPZyGTJEkqmYVMkiSpZBYySZKkklnIJEmSSmYhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkq2QdkB1tbWW2+dgwYNKjuGJEnSGs2bN++FzOy7pvXWu0I2aNAg5s6dW3YMSZKkNYqIJ2pZz12WkiRJJbOQSZIklcxCJkmSVLL17hiy9ixfvpympiaWLVtWdhQVpHfv3vTv35/6+vqyo0iS1OG6RSFrampi0003ZdCgQURE2XHUwTKTpUuX0tTUxODBg8uOI0lSh+sWuyyXLVvGVlttZRnrpiKCrbbayhFQSVK3VVghi4iLIuK5iHhwFcsjIr4XEYsj4v6IGP4eX++9bK4uzs9XktSdFTlCdglw4GqWfxgYUv06ATivwCySJEldVmGFLDNvBV5czSoHA5dmxZ3AFhGxXUe9fp8+fVa7fNCgQTQ0NDBs2DAaGhr45S9/2VEvXZhLLrmEKVOmvGPeuHHjvFCuJEnruTIP6t8eeKrVdFN13jNtV4yIE6iMojFgwIAOC3DzzTez9dZbs2jRIiZMmMDBBx9c03aZSWbSq9ff++zKlSupq6vrsGxtrVixorDn7opWrFjBBht0i3NOJElaozIP6m/voKBsb8XM/FFmjszMkX37rvF2UO/wzDPP0NjYyLBhwxg6dCi33Xbbu9Z55ZVX2HLLLVumv/Od7zB06FCGDh3KOeecA8CSJUvYddddOfnkkxk+fDhPPfUUffr0Ydq0aYwZM4Y77riDG2+8kT333JOGhgaOOeYY3nzzTe6++24OPfRQAH75y1+y8cYb89Zbb7Fs2TJ23HFHAB577DEOPPBARowYwdixY3n44YcBOProozn11FMZP348p59++hrf6xVXXEFDQwNDhw59x/p9+vTh9NNPZ8SIEey///7cfffdjBs3jh133JFZs2YBlUI5depURo0axe67784Pf/jD5u89U6dOZejQoTQ0NDBz5kwATj755JZtJ02axDHHHAPAhRdeyFe+8pWW79fxxx/PBz/4QSZMmMAbb7zRoe9XkqTuoswhiCZgh1bT/YGnO/pFfvazn3HAAQfw5S9/mZUrV/L666+3LBs/fjyZyeOPP85VV10FwLx587j44ou56667yEzGjBnDvvvuy5ZbbsmiRYu4+OKL+cEPfgDAa6+9xtChQ5k+fTrLli1jyJAh3Hjjjeyyyy4ceeSRnHfeeUyZMoV7770XgNtuu42hQ4cyZ84cVqxYwZgxYwA44YQTOP/88xkyZAh33XUXJ598MjfddBMAjzzyCDfccAN1dXVccsklzJw5k9tvv73lPSxevBiAp59+mtNPP5158+ax5ZZbMmHCBK677joOOeQQXnvtNcaNG8dZZ53FpEmT+MpXvsL111/PggULOOqoo5g4cSIXXnghm2++OXPmzOHNN99kn332YcKECdxzzz3Mnz+f++67jxdeeIFRo0bR2NhIY2Mjt912GxMnTuQvf/kLzzxTGdi8/fbbmTx5MgCPPvooV1xxBT/+8Y/56Ec/ys9//nM++clP1vx+JUl6L0ZMvXSdtpt39pEdnGTNyixks4ApEXElMAZ4OTPftbvyvRo1ahTHHHMMy5cv55BDDmHYsGEty5p3WT722GPst99+jBs3jttvv51JkyaxySabAHDooYe2FI+BAwey1157tWxfV1fHYYcdBsCiRYsYPHgwu+yyCwBHHXUUM2bM4HOf+xw777wzCxcu5O677+bUU0/l1ltvZeXKlYwdO5ZXX32VP/3pTxx++OEtz/vmm2+2PD788MPfUU4+9rGPce6557ZMjxs3DoA5c+Ywbtw4mkcQP/GJT3DrrbdyyCGHsOGGG3LggZXzKxoaGthoo42or6+noaGBJUuWAPD73/+e+++/n2uuuQaAl19+mUcffZTbb7+dI444grq6Ovr168e+++7LnDlzGDt2LOeccw4LFixgt91246WXXuKZZ57hjjvu4Hvf+x5Lly5l8ODBLd/vESNGsGTJkrV+v5Ik9QSFFbKIuAIYB2wdEU3AV4F6gMw8H5gNHAQsBl4HPl1EjsbGRm699VZ+/etf86lPfYqpU6dy5JHvbL477bQT/fr1Y8GCBWS2u9cUoKWkNevdu3dLeVjddmPHjuU3v/kN9fX17L///hx99NGsXLmSb33rW7z99ttsscUWzJ8/v6bXXJXVvX59fX3LZSN69erFRhtt1PK4+di0zOT73/8+BxxwwDu2nT17drvPuf322/PSSy/x29/+lsbGRl588UWuuuoq+vTpw6abbsrSpUtbXgcq5fWNN97osPcrSVJ3UuRZlkdk5naZWZ+Z/TPzwsw8v1rGqJ5d+a+ZuVNmNmRmIacKPvHEE2yzzTYcf/zxHHvssdxzzz3vWue5557jz3/+MwMHDqSxsZHrrruO119/nddee41rr72WsWPHrvF1PvCBD7BkyZKWXYiXXXYZ++67L1Apheeccw577703ffv2ZenSpTz88MN88IMfZLPNNmPw4MFcffXVQKUY3XfffWv9PseMGcMf/vAHXnjhBVauXMkVV1zR8vq1OOCAAzjvvPNYvnw5UNl1+Nprr9HY2MjMmTNZuXIlzz//PLfeeiujR48GYO+99+acc86hsbGRsWPH8q1vfWuN36uOer+SJHUn3f40tltuuYWzzz6b+vp6+vTpw6WX/n1/8vjx46mrq2P58uV84xvfoF+/fvTr14+jjz66pXQcd9xx7Lnnni279lald+/eXHzxxRx++OGsWLGCUaNGceKJJwKVsvTss8/S2NgIwO67784222zTMmp1+eWXc9JJJ/G1r32N5cuXM3nyZPbYY4+1ep/bbbcdZ555ZstxcQcddFDNZ402v88lS5YwfPhwMpO+ffty3XXXMWnSJO644w722GMPIoJvfvObbLvttkBl5O/3v/89O++8MwMHDuTFF1+sqbx2xPuVJKk7idXt6uqKRo4cmW2vu7Vw4UJ23XXXkhKps/g5S5LWRlc4qD8i5mXmyDWt1y3uZSlJkrQ+s5BJkiSVzEImSZJUMguZJElSySxkkiRJJbOQSZIklazbX4esM63r6bWrUstpt3V1dTQ0NLRMX3fddQwaNKjddZcsWcJHPvIRHnzwwY6KKEmSOoCFbD238cYbr/I2RJIkaf3gLstuaMmSJYwdO5bhw4czfPhw/vSnP71rnYceeojRo0czbNgwdt99dx599FEAfvrTn7bM/8xnPsPKlSs7O74kST2OhWw998YbbzBs2DCGDRvGpEmTANhmm224/vrrueeee5g5cyannHLKu7Y7//zz+exnP8v8+fOZO3cu/fv3Z+HChcycOZM//vGPzJ8/n7q6Oi6//PLOfkuSJPU47rJcz7W3y3L58uVMmTKlpVQ98sgj79pu77335utf/zpNTU0ceuihDBkyhBtvvJF58+YxatQooFL2ttlmm055H5Ik9WQWsm7ou9/9Lv369eO+++7j7bffpnfv3u9a5+Mf/zhjxozh17/+NQcccAAXXHABmclRRx3FmWeeWUJqSZJ6LndZdkMvv/wy2223Hb169eKyyy5r9ziwxx9/nB133JFTTjmFiRMncv/997PffvtxzTXX8NxzzwHw4osv8sQTT3R2fEmSehxHyDpQR94d/r04+eSTOeyww7j66qsZP348m2yyybvWmTlzJj/96U+pr69n2223Zdq0abzvfe/ja1/7GhMmTODtt9+mvr6eGTNmMHDgwBLehSRJPUdkZtkZ1srIkSNz7ty575i3cOFCdt1115ISqbP4OUuS1sa6Xh+0IwdYImJeZo5c03ruspQkSSqZhUySJKlkFjJJkqSSWcgkSZJKZiGTJEkqmYVMkiSpZF6HrAM9Ob2hQ59vwLQHVrt86dKl7LfffgD89a9/pa6ujr59+wJw9913s+GGG3ZoHkmSVAwL2Xpsq622armP5RlnnEGfPn047bTT3rFOZpKZ9OrlYKgkSV2Vv6W7ocWLFzN06FBOPPFEhg8fzlNPPcUWW2zRsvzKK6/kuOOOA+DZZ5/l0EMPZeTIkYwePZo777yzrNiSJPVYFrJuasGCBRx77LHce++9bL/99qtc75RTTuELX/gCc+fO5aqrrmopapIkqfO4y7Kb2mmnnRg1atQa17vhhhtYtGhRy/RLL73EG2+8wcYbb1xkPEmS1IqFrJtqfUPxXr160fqepcuWLWt5nJmeACBJUsncZdkD9OrViy233JJHH32Ut99+m2uvvbZl2f7778+MGTNapptPEpAkSZ3HEbIOtKbLVJTprLPO4sADD2TAgAHstttuvPnmmwDMmDGDk046iYsvvpgVK1Ywfvz4dxQ0SZJUvGi9K2t9MHLkyJw7d+475i1cuJBdd921pETqLH7OkqS1MWLqpeu03byzj+ywDBExLzNHrmk9d1lKkiSVzEImSZJUsm5TyNa3Xa9aO36+kqTurFsUst69e7N06VJ/aXdTmcnSpUvp3bt32VEkSSpEtzjLsn///jQ1NfH888+XHUUF6d27N/379y87hiRJhegWhay+vp7BgweXHUOSJGmddItdlpIkSeszC5kkSVLJusUuS0mSVJ51vQArdOxFWNdnjpBJkiSVzEImSZJUskILWUQcGBGLImJxRHyxneUDIuLmiLg3Iu6PiIOKzCNJktQVFVbIIqIOmAF8GNgNOCIidmuz2leAqzJzT2Ay8IOi8kiSJHVVRY6QjQYWZ+bjmfkWcCVwcJt1Etis+nhz4OkC80iSJHVJRRay7YGnWk03Vee1dgbwyYhoAmYD/9beE0XECRExNyLmejV+SZLU3RRZyKKdeW1vNnkEcElm9gcOAi6LiHdlyswfZebIzBzZt2/fAqJKkiSVp8hC1gTs0Gq6P+/eJXkscBVAZt4B9Aa2LjCTJElSl1NkIZsDDImIwRGxIZWD9me1WedJYD+AiNiVSiFzn6QkSepRCitkmbkCmAL8DlhI5WzKhyJiekRMrK7278DxEXEfcAVwdGa23a0pSZLUrRV666TMnE3lYP3W86a1erwA2KfIDJIkSV2dV+qXJEkqmYVMkiSpZBYySZKkklnIJEmSSmYhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkpW6IVhJUnr7snpDeu87YBpD3RgEklFc4RMkiSpZBYySZKkkrnLUpKk9ciIqZeu87bzzj6yA5OoIzlCJkmSVDILmSRJUsksZJIkSSWzkEmSJJXMg/oldTqvryVJ7+QImSRJUskcIZMkaRW8xIQ6iyNkkiRJJbOQSZIklcxCJkmSVDILmSRJUsksZJIkSSWzkEmSJJXMQiZJklQyC5kkSVLJLGSSJEkls5BJkiSVzEImSZJUMguZJElSySxkkiRJJbOQSZIklcxCJkmSVDILmSRJUsksZJIkSSWzkEmSJJXMQiZJklQyC5kkSVLJLGSSJEkls5BJkiSVzEImSZJUMguZJElSyTYo8skj4kDg/wJ1wAWZ+Y121vkocAaQwH2Z+fEiM0k9zZPTG9Z52wHTHujAJJKkVampkEXELsBUYGDrbTLzQ6vZpg6YAfwz0ATMiYhZmbmg1TpDgC8B+2TmSxGxzTq9C0mSpPVYrSNkVwPnAz8GVta4zWhgcWY+DhARVwIHAwtarXM8MCMzXwLIzOdqfG5JkqRuo9ZCtiIzz1vL594eeKrVdBMwps06uwBExB+p7NY8IzN/2/aJIuIE4ASAAQMGrGUMSZKkrq3Wg/r/OyJOjojtIuJ9zV9r2CbamZdtpjcAhgDjgCOACyJii3dtlPmjzByZmSP79u1bY2RJkqT1Q60jZEdV/zu11bwEdlzNNk3ADq2m+wNPt7POnZm5HPhzRCyiUtDm1JhLkiRpvVdTIcvMwevw3HOAIRExGPgLMBloewbldVRGxi6JiK2p7MJ8fB1eS5Ikab1V61mW9cBJQGN11i3AD6sjW+3KzBURMQX4HZXjwy7KzIciYjowNzNnVZdNiIgFVE4WmJqZS9f53UiSJK2Hat1leR5QD/ygOv2p6rzjVrdRZs4GZreZN63V4wROrX5JkiT1SLUWslGZuUer6Zsi4r4iAkmSJPU0tRaylRGxU2Y+BhARO1L79ch6nBFTL13nbeedfWQHJpEkSeuDWgvZVODmiHicyuUsBgKfLiyVJElSD1LrWZY3Vm9z9H4qhezhzHyz0GSSJEk9xGoLWUR8KDNviohD2yzaKSLIzF8UmE2SJKlHWNMI2b7ATcD/amdZAhYySZKk92i1hSwzv1p9OD0z/9x6WfWCr5KqnpzesM7bDpj2QAcm0brw85NUplrvZfnzduZd05FBJEmSeqo1HUP2AeCDwOZtjiPbDOhdZDB1LC/FIUlS17WmY8jeD3wE2IJ3Hkf2N+D4okJJkiT1JGs6huyXEfEr4PTM/K9OyiRJktSjrPE6ZJm5MiL+GbCQSVIP58kPUjFqvVL/nyLiXGAm8FrzzMy8p5BUkiRJPUithewfq/+d3mpeAh/q2DiSJEk9T623ThpfdBBJkqSeqqbrkEXE5hHxnYiYW/36dkRsXnQ4SZKknqDWC8NeROVSFx+tfr0CXFxUKEmSpJ6k1mPIdsrMw1pN/0dEzC8ikCRJUk9TayF7IyL+KTNvB4iIfYA3ioslrZ6n3kvdj3cUUU9WayE7CfhJ9bixAF4EjioslSRJUg9S61mW84E9ImKz6vQrhaZaC/5FJUmS1nc1FbKI2Ar4KvBPQEbE7cD0zFxaZLiiudtLkiR1BbWeZXkl8DxwGPAv1ccziwolSZLUk9R6DNn7MvM/W01/LSIOKSKQeo73srv52k07MIik9Z57PLS+q3WE7OaImBwRvapfHwV+XWQwSZKknqLWQvYZ4GfAW9WvK4FTI+JvEdFlDvCXJElaH9V6lqU7iCRJkgpS6zFkRMREoLE6eUtm/qqYSJIkST1LrTcX/wbwWWBB9euz1XmSJEl6j2odITsIGJaZbwNExE+Ae4EvFhVMkiSpp6h5lyWwBZVbJgFsXkAWSVInWdfLznjJGakYtRayM4F7I+JmKveybAS+VFgqSZKkHmSNhSwiArgd2AsYRaWQnZ6Zfy04myRJUo+wxkKWmRkR12XmCGBWJ2RSF+MVsCVJKlatF4a9MyJGFZpEkiSph6r1GLLxwIkRsQR4jcpuy8zM3YsKJpXFg53XX94fVdL6qtZC9uFCU0iSJPVgqy1kEdEbOBHYGXgAuDAzV3RGMEmSpJ5iTceQ/QQYSaWMfRj4duGJJEmSepg17bLcLTMbACLiQuDu4iNJkiT1LGsqZMubH2TmisolyVQkLzEhSVLPs6ZCtkdEvFJ9HMDG1enmsyw3KzSdJElSD7DaQpaZdZ0VRJIkqaeq9cKw6yQiDoyIRRGxOCK+uJr1/iUiMiJGFplHkiSpKyqskEVEHTCDytmZuwFHRMRu7ay3KXAKcFdRWSRJkrqyIkfIRgOLM/PxzHwLuBI4uJ31/hP4JrCswCySJEldVpGFbHvgqVbTTdV5LSJiT2CHzPxVgTkkSZK6tCILWXvXyMiWhRG9gO8C/77GJ4o4ISLmRsTc559/vgMjSpIkla/IQtYE7NBquj/wdKvpTYGhwC3Vm5bvBcxq78D+zPxRZo7MzJF9+/YtMLIkSVLnK7KQzQGGRMTgiNgQmAzMal6YmS9n5taZOSgzBwF3AhMzc26BmSRJkrqcNV0Ydp1Vr+w/BfgdUAdclJkPRcR0YG5mzlr9M0hqNmLqpeu87bWbdmAQSVIhCitkAJk5G5jdZt60Vaw7rsgskiRJXVWhF4aVJEnSmlnIJEmSSmYhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkpmIZMkSSqZhUySJKlkFjJJkqSSWcgkSZJKZiGTJEkqmYVMkiSpZBYySZKkklnIJEmSSmYhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkpmIZMkSSqZhUySJKlkFjJJkqSSWcgkSZJKZiGTJEkq2QZlB5AkqTt6cnrDOm87YNoDHZhE6wNHyCRJkkrmCJmkdTJi6qXrvO21m3ZgEEnqBhwhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkrmQf2SJPUQXoqj63KETJIkqWQWMkmSpJK5y1KSCuT12iTVwhEySZKkklnIJEmSSmYhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkpmIZMkSSqZhUySJKlkFjJJkqSSFVrIIuLAiFgUEYsj4ovtLD81IhZExP0RcWNEDCwyjyRJUldUWCGLiDpgBvBhYDfgiIjYrc1q9wIjM3N34Brgm0XlkSRJ6qqKHCEbDSzOzMcz8y3gSuDg1itk5s2Z+Xp18k6gf4F5JEmSuqQiC9n2wFOtppuq81blWOA3BeaRJEnqkjYo8LmjnXnZ7ooRnwRGAvuuYvkJwAkAAwYM6Kh8kiRJXUKRI2RNwA6tpvsDT7ddKSL2B74MTMzMN9t7osz8UWaOzMyRffv2LSSsJElSWYosZHOAIRExOCI2BCYDs1qvEBF7Aj+kUsaeKzCLJElSl1VYIcvMFcAU4HfAQuCqzHwoIqZHxMTqamcDfYCrI2J+RMxaxdNJkiR1W0UeQ0ZmzgZmt5k3rdXj/Yt8fUmSpPWBV+qXJEkqWaEjZJIkSavz5PSGdd52wLQHOjBJuRwhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkpmIZMkSSqZhUySJKlkFjJJkqSSWcgkSZJKZiGTJEkqmYVMkiSpZBYySZKkklnIJEmSSmYhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkpmIZMkSSqZhUySJKlkFjJJkqSSWcgkSZJKZiGTJEkqmYVMkiSpZBYySZKkklnIJEmSSmYhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkpmIZMkSSqZhUySJKlkFjJJkqSSWcgkSZJKZiGTJEkqmYVMkiSpZBYySZKkklnIJEmSSmYhkyRJKpmFTJIkqWQWMkmSpJJZyCRJkkpmIZMkSSpZoYUsIg6MiEURsTgivtjO8o0iYmZ1+V0RMajIPJIkSV1RYYUsIuqAGcCHgd2AIyJitzarHQu8lJk7A98FzioqjyRJUle1QYHPPRpYnJmPA0TElcDBwIJW6xwMnFF9fA1wbkREZmaBuSRJklbpyekN67ztgGkPrNN2Re6y3B54qtV0U3Veu+tk5grgZWCrAjNJkiR1OVHUYFREHA4ckJnHVac/BYzOzH9rtc5D1XWaqtOPVddZ2ua5TgBOqE6+H1jUQTG3Bl7ooOfqKGaqjZlq1xVzmak2ZqpdV8xlptp090wDM7PvmlYqcpdlE7BDq+n+wNOrWKcpIjYANgdebPtEmfkj4EcdHTAi5mbmyI5+3vfCTLUxU+26Yi4z1cZMteuKucxUGzNVFLnLcg4wJCIGR8SGwGRgVpt1ZgFHVR//C3CTx49JkqSeprARssxcERFTgN8BdcBFmflQREwH5mbmLOBC4LKIWExlZGxyUXkkSZK6qiJ3WZKZs4HZbeZNa/V4GXB4kRnWoMN3g3YAM9XGTLXrirnMVBsz1a4r5jJTbcxEgQf1S5IkqTbeOkmSJKlkFjJJkqSSdftCFhEXRcRzEfHgKpZHRHyvej/N+yNieCdk2iEibo6IhRHxUER8tuxcEdE7Iu6OiPuqmf6jnXVKufdoRNRFxL0R8auukCkilkTEAxExPyLmtrO8jJ+pLSLimoh4uPpztXcXyPT+6veo+euViPhc2blWkXVSRGREfKCM119dhoj4fEQsi4jNy8pWzbGy+jneFxH3RMQ/lpmnWURsGxFXRsRjEbEgImZHxC4l5mn+Pj1U/V6dGhGl/65tlav56133l+6EDK+2mT46Is6tPj4jIv5SzbYgIo7opEwZEd9uNX1aRJzRavrIiHiw+nkuiIjTCguTmd36C2gEhgMPrmL5QcBvgAD2Au7qhEzbAcOrjzcFHgF2KzNX9XX6VB/XA3cBe7VZ52Tg/OrjycDMTvoMTwV+BvyqnWWdnglYAmy9muVl/Ez9BDiu+nhDYIuyM7V5/Trgr1QukNhlcrXKcRVwG3BGGa+/ugzA3dX5R5eVrZrj1VaPDwD+UGaeao4A7gBObDVvGDC2i3yftgFuAP6jC3yvXu1qGYCjgXOrj88ATqs+HgK8AtR3QqZlwJ+b/00HTmv+f5DKvbjvAf6hOt0bOL6oLKW39qJl5q20c7HZVg4GLs2KO4EtImK7gjM9k5n3VB//DVjIu28r1am5qq/T/NdLffWr7RkfB1P5xQ+Ve4/uFxFRVCaAiOgP/E/gglWs0umZatCpn11EbEblD48LATLzrcz8f2Vmasd+wGOZ+UQXy0VE9AH2AY6lpEvvrCpDROwE9AG+AnTKiEGNNgNeKjsEMB5YnpnnN8/IzPmZeVuJmVpk5nNU7jIzpQv8u7TeyMxHgdeBLTvh5VZQOaPy8+0s+xKVkvh0NdeyzPxxUUG6fSGrQS333CxMdRfbnlRGpFrr9FzVXYPzgeeA6zNzlZmy8+49eg7wBeDtVSwvI1MCv4+IeVG5rdcqM1UV/dntCDwPXFzdtXtBRGxScqa2JgNXtDO/7FwAhwC/zcxHgBdL2m26qgxHUPm+3Qa8PyK2KSFbs42ru5MepvIH0n+WmKXZUGBe2SFWJzMfp/K7tszPDv7++TV/fazsDMD09laq/vw/Wi20nWEG8Il2Dgvo1J8vC1llyLutTrkWSPWv4p8Dn8vMV9oubmeTQnNl5srMHEblNlejI2JomZki4iPAc5m5uv8hyvj89snM4VSGs/81IhpLzrQBld3y52XmnsBrQNvjQ8r8Od8vAVTsAAAE9ElEQVQQmAhc3d7iduZ19rV4jgCurD6+knJGolaVYTJwZWa+DfyCcq/b+EZmDsvMDwAHApc66lOzrvB9av78mr9mlp0BmNZm+ecjYhGVAYozOitU9ffvpcApnfWa7bGQ1XbPzQ4XEfVUytjlmfmLrpILoLq76xYq/+i2mylWc+/RDrQPMDEillD5JfWhiPhpyZloNXz9HHAtMHpVmaqK/uyagKZWI5rXUCloZWZq7cPAPZn5bDvLysxFRGwFfAi4oPpzNhX4WGcWjdVk2IPKsTTXV+dPpovstszMO6jcfHmNN0wu2EPAiJIzrFZE7AispLLnQav33cx8P/AxKoW/dye+9jlUDhlovXehU3++LGSV+2keWT3bay/g5cx8psgXrP5jfyGwMDO/0xVyRUTfiNii+nhjYH/g4XYyddq9RzPzS5nZPzMHUflldFNmfrLMTBGxSURs2vwYmAC0PYO3Uz+7zPwr8FREvL86az9gQZmZ2mje7daeMnNB5Wfm0swcmJmDMnMHKgf4/lMXyHAOlYOLB1W//gHYPiIGdmK2dkXlTNA6YGnJUW4CNoqI45tnRMSoiNi3xEwtIqIvcD6VA9e9CnuNqoMUc/n7v+2d8ZovUjmx5thWs88EvhkR20LLWf2FjaIVeuukriAirgDGAVtHRBPwVSoHrFM9EHQ2lTO9FlM5iPDTnRBrH+BTwAPV/egA/xsYUGKu7YCfREQdlaJ+VWb+KrrgvUdLztQPuLY6gLIB8LPM/G1EnAil/kz9G3B5dffg48Cnu0AmIuJ/AP8MfKbVvNJztXIE8I02834OfJzKcVtlZvg8lRHY1q6l8jN+VifkamvjVv9eBXBUZq4sIUeLzMyImAScE5XLOCyjchb051a7YbGav0/1VA4YvwxY1R/enan15weVYxY7/dIXa2E68LOI+HF1l31n+DYwpXkiM2dHRD/ghupASgIXFfXi3jpJkiSpZO6ylCRJKpmFTJIkqWQWMkmSpJJZyCRJkkpmIZMkSSqZhUxStxERK6u3ZXkwIv67+dp6q1l/i4g4udX0P0TENcUnlaR38rIXkrqNiHg1M/tUH/8EeCQzv76a9QcBv8rMtrcJk6RO5QiZpO7qDqo3Ko+IPhFxY0TcExEPRMTB1XW+AexUHVU7OyIGRcSD1W2OjohfRMRvI+LRiPhm8xNHxLER8UhE3BIRP46Iczv93UnqVrr9lfol9TzVO07sR+VODlC5gvukzHwlIrYG7oyIWVRuwj60eqPj5hGz1oYBewJvAosi4vtU7kv4f6jcL/RvVG7fc1+hb0hSt2chk9SdNN8eZhAwD7i+Oj+A/4qIRuBtKiNn/Wp4vhsz82WAiFgADKRyU+0/VO99R0RcDezSkW9CUs/jLktJ3ckb1dGugcCGwL9W538C6AuMqC5/Fuhdw/O92erxSip/xEbHxZWkCguZpG6nOqp1CnBaRNQDmwPPZebyiBhPpbBBZZfjpmv59HcD+0bElhGxAXBYR+WW1HNZyCR1S5l5L5VjuyYDlwMjI2IuldGyh6vrLAX+WL1Mxtk1Pu9fgP8C7gJuABYAL3f8O5DUk3jZC0laSxHRJzNfrY6QXQtclJnXlp1L0vrLETJJWntnVE8eeBD4M3BdyXkkreccIZMkSSqZI2SSJEkls5BJkiSVzEImSZJUMguZJElSySxkkiRJJfv/NoVO4HBLBeQAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#get proportions\n",
    "df_rating_home['Proportion'] = df_rating_home['ListingNumber_x']/df_rating_home['ListingNumber_y']\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = df_rating_home, x = 'Rating', y = 'Proportion', hue = 'IsBorrowerHomeowner');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> We can see a peak of homeowners in the centre, where rating is AA/A or 6/7, decreasing with lower credit ratings. The opposite trend can be observed for non homeowners.  \n",
    "\n",
    "### 5) Homeowner vs Loan Status\n",
    "> Do homeowners tend to default or have loans charged off less often?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "default_off = df_loans[df_loans.LoanStatus.isin(['Chargedoff', 'Defaulted'])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 17000 entries, 74792 to 106807\n",
      "Data columns (total 15 columns):\n",
      "ListingNumber                17000 non-null int64\n",
      "ListingCreationDate          17000 non-null datetime64[ns]\n",
      "CreditGrade                  10660 non-null object\n",
      "Term                         17000 non-null int64\n",
      "LoanStatus                   17000 non-null object\n",
      "ClosedDate                   17000 non-null datetime64[ns]\n",
      "ProsperRating (numeric)      6340 non-null float64\n",
      "ProsperScore                 6340 non-null float64\n",
      "ListingCategory (numeric)    17000 non-null int64\n",
      "Occupation                   16187 non-null object\n",
      "EmploymentStatus             16187 non-null object\n",
      "IsBorrowerHomeowner          17000 non-null bool\n",
      "Duration                     17000 non-null float64\n",
      "months                       17000 non-null float64\n",
      "Rating                       17000 non-null object\n",
      "dtypes: bool(1), datetime64[ns](2), float64(4), int64(3), object(5)\n",
      "memory usage: 2.0+ MB\n"
     ]
    }
   ],
   "source": [
    "default_off.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ListingNumber</th>\n",
       "      <th>ListingCreationDate</th>\n",
       "      <th>CreditGrade</th>\n",
       "      <th>Term</th>\n",
       "      <th>LoanStatus</th>\n",
       "      <th>ClosedDate</th>\n",
       "      <th>ProsperRating (numeric)</th>\n",
       "      <th>ProsperScore</th>\n",
       "      <th>ListingCategory (numeric)</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>EmploymentStatus</th>\n",
       "      <th>Duration</th>\n",
       "      <th>months</th>\n",
       "      <th>Rating</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>IsBorrowerHomeowner</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>False</th>\n",
       "      <td>9400</td>\n",
       "      <td>9400</td>\n",
       "      <td>5998</td>\n",
       "      <td>9400</td>\n",
       "      <td>9400</td>\n",
       "      <td>9400</td>\n",
       "      <td>3402</td>\n",
       "      <td>3402</td>\n",
       "      <td>9400</td>\n",
       "      <td>8793</td>\n",
       "      <td>8793</td>\n",
       "      <td>9400</td>\n",
       "      <td>9400</td>\n",
       "      <td>9400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>True</th>\n",
       "      <td>7600</td>\n",
       "      <td>7600</td>\n",
       "      <td>4662</td>\n",
       "      <td>7600</td>\n",
       "      <td>7600</td>\n",
       "      <td>7600</td>\n",
       "      <td>2938</td>\n",
       "      <td>2938</td>\n",
       "      <td>7600</td>\n",
       "      <td>7394</td>\n",
       "      <td>7394</td>\n",
       "      <td>7600</td>\n",
       "      <td>7600</td>\n",
       "      <td>7600</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     ListingNumber  ListingCreationDate  CreditGrade  Term  \\\n",
       "IsBorrowerHomeowner                                                          \n",
       "False                         9400                 9400         5998  9400   \n",
       "True                          7600                 7600         4662  7600   \n",
       "\n",
       "                     LoanStatus  ClosedDate  ProsperRating (numeric)  \\\n",
       "IsBorrowerHomeowner                                                    \n",
       "False                      9400        9400                     3402   \n",
       "True                       7600        7600                     2938   \n",
       "\n",
       "                     ProsperScore  ListingCategory (numeric)  Occupation  \\\n",
       "IsBorrowerHomeowner                                                        \n",
       "False                        3402                       9400        8793   \n",
       "True                         2938                       7600        7394   \n",
       "\n",
       "                     EmploymentStatus  Duration  months  Rating  \n",
       "IsBorrowerHomeowner                                              \n",
       "False                            8793      9400    9400    9400  \n",
       "True                             7394      7600    7600    7600  "
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "default_off.groupby('IsBorrowerHomeowner').count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5529411764705883"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "9400/17000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True     56440\n",
       "False    55910\n",
       "Name: IsBorrowerHomeowner, dtype: int64"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.IsBorrowerHomeowner.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.4976412995104584"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "55910/(55910+56440)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> While in the full dataset we have that 49.8% of the borrowers are not homeowners, when we look at the population of charged off and defaulted loans, 55.3% do not own a house, indicating there might be some correlation between these two variables. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6) Duration vs Term\n",
    "> We have calculated the duration in months for the completed loans, as we do not have a closed date for Current or Past Due loans. We now want to check the performance between duration and originally set term of loans that are completed. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_duration = df_loans[df_loans['months'].notnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Ky4bw7//OUjkdpeKGe6j4OCLLF+xghtvvNHpckQkxymEiUhOW7hwITf9/d/j7WmkaMcjuAYOO13S8azF27ad4u2/JuBxcdutt+J2u52uSmSUuiNzk0KYiOS897znPXzpS1+iImAo3PEIvqYNOdMqZob7KHj9NwQa13PO2au49957mD9/vtNliRxFISw3aYkKEZkWLrjgAn5w373cdddd/OY3v8Hf08jgvPOIl9Y6U1Aijq91K4HW1wj4fXz005/m2muvxRjjTD0ib0IhLDcphInItFFcXMytt97KFVdcwVfuvJPWXb8jWr6QcP0arK8wa3W4ew9RcHA9DPVy8aWX8tGbb2b27NlZu77IRGmJitykECYi086aNWu47957+elPf8oPf/hDfFubGJp7FtHqlZDBligTHcJ/4AW83fupmVvLLZ//f1izZk3GrieSLgphuUkhTESmJb/fz9/8zd9w1VVX8bWvfY3169fjCbUytPCSjOwn6epvp3DvU3jiYT7woQ/xnve8B5/Pl/briGSCQlhu0sB8EZnWampq+NKXvsTHPvYxfH2HKN7xCK7BzvRdIDXzsfD1x6kuL+Lb376L97///QpgMq1Eo1GnS5ATUAgTkWnPGMONN97I17/+NSoKvRTtfAxXf0dazu079CqBxvVccN55/O/3vseSJUvScl6RbBgZjh+JRBytQ05MIUxEZozTTjuN//3e95hVWUHh3ienvMq+p2s//pbXePvb384dd9xBcXFxmioVyY6RWZHhcNjhSuREFMJEZEapqKjgjttvx5sIE9z7NExyar4Z7qVg/7MsXbqMT3ziE7hceruU6Wfk5a+WsNykdxURmXFOPfVUbr75Ztx9zbj7mid1Dn/za/i8bv7t376g8V8ybY2MBVNLWG5SCBORGenaa6+lqLgYb8frxz3mb1yPe7AT92AnwZ2P429cf/QBsTC+nv1cfdVVWv9LprVweBiA4eFhhyuRE1EIE5EZyefz8fZrr8Xb0wixo7tiXINdmHgUE4/iCbXiGuw66nFv935sPMa6deuyWbJI2o20gCmE5SaFMBGZsc455xywCdxDXSc/+AiugcMUFBZx6qmnZqgykcxLJBJEU2PBFMJyk0KYiMxYCxcuBDiupetkPEPdLF68SPtAyrR2ZPBSCMtNCmEiMmNVVVXh9wdwhUMTep470s/8+voMVSWS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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "sb.violinplot(data = df_duration, x = 'Term', y = 'months', color = base);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Term  LoanStatus            \n",
       "12    Chargedoff                   72\n",
       "      Completed                  1449\n",
       "      Current                      62\n",
       "      Defaulted                    10\n",
       "      FinalPaymentInProgress       10\n",
       "      Past Due (1-15 days)          3\n",
       "      Past Due (16-30 days)         3\n",
       "      Past Due (31-60 days)         1\n",
       "      Past Due (61-90 days)         2\n",
       "      Past Due (91-120 days)        1\n",
       "36    Cancelled                     5\n",
       "      Chargedoff                10828\n",
       "      Completed                 34058\n",
       "      Current                   35362\n",
       "      Defaulted                  4818\n",
       "      FinalPaymentInProgress      155\n",
       "      Past Due (1-15 days)        553\n",
       "      Past Due (16-30 days)       174\n",
       "      Past Due (31-60 days)       239\n",
       "      Past Due (61-90 days)       205\n",
       "      Past Due (91-120 days)      193\n",
       "      Past Due (>120 days)          9\n",
       "60    Chargedoff                 1086\n",
       "      Completed                  2414\n",
       "      Current                   19733\n",
       "      Defaulted                   186\n",
       "      FinalPaymentInProgress       37\n",
       "      Past Due (1-15 days)        249\n",
       "      Past Due (16-30 days)        88\n",
       "      Past Due (31-60 days)       123\n",
       "      Past Due (61-90 days)       105\n",
       "      Past Due (91-120 days)      110\n",
       "      Past Due (>120 days)          7\n",
       "Name: ListingNumber, dtype: int64"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.groupby(['Term', 'LoanStatus']).count()['ListingNumber']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Timestamp('2010-07-02 02:01:21.567000')"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans[df_loans['Term']==60]['ListingCreationDate'].min()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Timestamp('2014-03-10 12:20:53.760000')"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans['ListingCreationDate'].max()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> A very interesting insight can be gathered here: most of the loans with term of 12 or 36 months were kept across the whole duration, whereas the ones with a term of 5 years were repaid earlier. It's also true that we do not have much data as most of the loans with a term of 60 months are still active. The first of these loans was created in 2010 and the dataset gets to 2014. \n",
    "\n",
    "### 7) Duration vs Loan Status\n",
    "> Let's look at the term distribution of charged off and defaulted loans."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(data = default_off, x = 'months'); #right-skewed distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#on the whole dataset for comparison\n",
    "plt.hist(data = df_loans, x = 'months');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> We can see the peak is in 1Y duration rather than 3Y as in the full dataset. We will look now at the term. \n",
    "\n",
    "### 8) Term vs Loan Status"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "total_default_off = default_off.shape[0]\n",
    "\n",
    "ax_default_off = sb.countplot(data = default_off, x = 'Term', color = base)\n",
    "\n",
    "for p in ax_default_off.patches:\n",
    "    height = p.get_height()\n",
    "    ax_default_off.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total_default_off),\n",
    "            ha=\"center\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#on the whole dataset for comparison\n",
    "ax = sb.countplot(data = df_loans, x = 'Term', color= base)\n",
    "total = df_loans.shape[0]\n",
    "\n",
    "for p in ax.patches:\n",
    "    height = p.get_height()\n",
    "    ax.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:1.2f}'.format(height/total),\n",
    "            ha=\"center\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Let's look at whether the creation or closed date can be an indicator of default."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "default_year_create = pd.DataFrame(default_off.set_index('ListingCreationDate').groupby(pd.Grouper(freq='Y'))['ListingNumber'].count().reset_index())\n",
    "default_year_create['Proportion'] = default_year_create['ListingNumber']/total_default_off"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = default_year_create, x = 'ListingCreationDate', y= 'Proportion', color = base)\n",
    "plt.xticks(plt.xticks()[0], (default_year_create.ListingCreationDate.dt.year.astype(str)), rotation=90)\n",
    "plt.tight_layout();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "create = pd.DataFrame(df_loans.set_index('ListingCreationDate').groupby(pd.Grouper(freq='Y'))['ListingNumber'].count().reset_index())\n",
    "create['Proportion'] = create['ListingNumber']/total"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = create, x = 'ListingCreationDate', y= 'Proportion', color = base)\n",
    "\n",
    "plt.xticks(plt.xticks()[0], (create.ListingCreationDate.dt.year.astype(str)), rotation=90)\n",
    "plt.tight_layout();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "default_year_close = pd.DataFrame(default_off.set_index('ClosedDate').groupby(pd.Grouper(freq='Y'))['ListingNumber'].count().reset_index())\n",
    "default_year_close['Proportion'] = default_year_close['ListingNumber']/total_default_off"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAsgAAAFgCAYAAACmDI9oAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAHO5JREFUeJzt3X28bXVdJ/DPt4tPZZIgmfGs4gM+YVzQsjHxEaqRmtBQC2xSe5DJGV9D2jQvNZocHV496KQpJgpNhkqlt8SIEbEp07ggooAkksoNNQrKStKufuePvW/8vB72PhfPvvvcc9/v12u/7l6/tdY+3/t9nX3256zzW2tVdwcAAJj4hmUXAAAA64mADAAAAwEZAAAGAjIAAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAwT7LLmCt3Ote9+rDDjts2WUAALBOXXbZZX/b3QfM227DBOTDDjssW7duXXYZAACsU1X1qdVsZ4oFAAAMBGQAABgIyAAAMBCQAQBgICADAMBAQAYAgIGADAAAAwEZAAAGAjIAAAwEZAAAGAjIAAAwEJABAGCwz7ILADaGo08/d9kl7BaXnXnKsksAYMEcQQYAgIGADAAAAwEZAAAGAjIAAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAYCMgAADARkAAAYCMgAADAQkAEAYCAgAwDAQEAGAICBgAwAAAMBGQAABgIyAAAMBGQAABgIyAAAMBCQAQBgICADAMBAQAYAgMFCA3JVHV9V11bVdVX14hXWv7Cqrq6qK6vqPVV16LDu1Kr6+PRx6iLrBACAHRYWkKtqU5LXJDkhyZFJnlFVR+602YeSbO7uhyc5P8n/mu67X5KXJnlUkmOTvLSq7rmoWgEAYIdFHkE+Nsl13X19d38pyXlJThw36O73dvcXposfSHLQ9PlTklzU3Td39y1JLkpy/AJrBQCAJIsNyAcmuWFY3jYduz0/nuTdu7JvVT2vqrZW1dabbrrp6ywXAAAWG5BrhbFeccOqH0myOcmZu7Jvd5/V3Zu7e/MBBxxwhwsFAIAdFhmQtyU5eFg+KMmNO29UVU9M8vNJntrdX9yVfQEAYK0tMiBfmuSIqjq8qu6c5OQkW8YNquqRSV6fSTj+m2HVhUmeXFX3nJ6c9+TpGAAALNQ+i3rh7t5eVadlEmw3JTm7u6+qqjOSbO3uLZlMqbh7krdXVZJ8uruf2t03V9UvZhKyk+SM7r55UbUCAMAOCwvISdLdFyS5YKexlwzPnzhj37OTnL246gAA4Gu5kx4AAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAYCMgAADARkAAAYCMgAADAQkAEAYCAgAwDAQEAGAICBgAwAAAMBGQAABgIyAAAMBGQAABgIyAAAMBCQAQBgICADAMBAQAYAgIGADAAAAwEZAAAGAjIAAAz2WXYBAABr4ejTz112CQt32ZmnLLuEvYIjyAAAMHAEGVbJkQkA2Ds4ggwAAAMBGQAABgIyAAAMBGQAABgIyAAAMBCQAQBgICADAMBAQAYAgIGADAAAAwEZAAAGAjIAAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAYCMgAADARkAAAYCMgAADAQkAEAYCAgAwDAQEAGAICBgAwAAIOFBuSqOr6qrq2q66rqxSusf2xVXV5V26vqpJ3Wfbmqrpg+tiyyTgAA2GGfRb1wVW1K8pokT0qyLcmlVbWlu68eNvt0kmcn+a8rvMSt3X3UouoDAICVLCwgJzk2yXXdfX2SVNV5SU5M8m8Bubs/OV33lQXWAQAAq7bIKRYHJrlhWN42HVutu1bV1qr6QFX9wEobVNXzpttsvemmm76eWgEAIMliA3KtMNa7sP8h3b05yTOT/FpV3e9rXqz7rO7e3N2bDzjggDtaJwAA/JtFBuRtSQ4elg9KcuNqd+7uG6f/Xp/kkiSPXMviAABgJYsMyJcmOaKqDq+qOyc5OcmqrkZRVfesqrtMn98ryWMyzF0GAIBFWVhA7u7tSU5LcmGSa5K8rbuvqqozquqpSVJVx1TVtiRPS/L6qrpquvuDk2ytqg8neW+SV+x09QsAAFiIRV7FIt19QZILdhp7yfD80kymXuy83/uTPGyRtQEAwErcSQ8AAAYCMgAADARkAAAYCMgAADAQkAEAYCAgAwDAQEAGAICBgAwAAAMBGQAABgIyAAAMBGQAABjss5qNquoBSU5Pcui4T3c/fkF1AQDAUqwqICd5e5LXJXlDki8vrhwAAFiu1Qbk7d39GwutBAAA1oHVzkH+g6r66aq6T1Xtt+Ox0MoAAGAJVnsE+dTpv6cPY53kvmtbDgAALNeqAnJ3H77oQgAAYD1Y7VUs7pTkp5I8djp0SZLXd/e/LqguAABYitVOsfiNJHdK8trp8o9Ox56ziKIAAGBZVhuQj+nuRwzLF1fVhxdREAAALNNqr2Lx5aq6346FqrpvXA8ZAIANaLVHkE9P8t6quj5JZXJHvR9bWFUAALAkq72KxXuq6ogkD8wkIH+su7+40MoAAGAJZgbkqnp8d19cVf9hp1X3q6p09+8tsDYAANjt5h1B/p4kFyf59yus6yQCMgAAG8rMgNzdL50+PaO7/2pcV1VuHgIAwIaz2qtY/O4KY+evZSEAALAezJuD/KAkD0my707zkO+R5K6LLAwAAJZh3hzkByb5/iTfkq+eh/yPSZ67qKIAAGBZ5s1BfmdV/WGSF3X3y3dTTQAAsDRz5yB395eTPGk31AIAAEu32jvpvb+qfj3JW5P8847B7r58IVUBAMCSrDYgf9f03zOGsU7y+LUtBwAAlmu1t5o+btGFAADAerCq6yBX1b5V9StVtXX6+OWq2nfRxQEAwO622huFnJ3Jpd2ePn18PsmbFlUUAAAsy2rnIN+vu39oWP6FqrpiEQUBAMAyrfYI8q1V9d07FqrqMUluXUxJAACwPKs9gvxTSc6ZzjuuJDcnOXVhVQEAwJKs9ioWVyR5RFXdY7r8+YVWBQB8laNPP3fZJewWl515yrJLgFVfxWL/qnp1kkuSvLeqXlVV+y+0MgAAWILVzkE+L8lNSX4oyUnT529dVFEAALAsq52DvF93/+Kw/D+q6gcWURAAACzTao8gv7eqTq6qb5g+np7kXYssDAAAlmG1AfknkrwlyZemj/OSvLCq/rGqnLAHAMCGsdqrWHzzogsB2MhcgQBgz7HaOcipqqcmeex08ZLu/sPFlAQAAMuz2su8vSLJC5JcPX28YDoGAAAbymqPIH9vkqO6+ytJUlXnJPlQkhcvqjAAAFiG1Z6klyTfMjzfd60LAQCA9WC1R5D/Z5IPVdV7k1Qmc5F/bmFVAQDAkswNyFVVSf40yaOTHJNJQH5Rd392wbUBAMBuN3eKRXd3knd092e6e0t3v3O14biqjq+qa6vquqr6mvnKVfXYqrq8qrZX1Uk7rTu1qj4+fZy66v8RAAB8HVY7B/kDVXXMrrxwVW1K8pokJyQ5MskzqurInTb7dJJnZ3ITknHf/ZK8NMmjkhyb5KVVdc9d+foAAHBHrDYgH5dJSP5EVV1ZVR+pqivn7HNskuu6+/ru3nH3vRPHDbr7k919ZZKv7LTvU5Jc1N03d/ctSS5KcvwqawUAgDtstSfpnXAHXvvAJDcMy9syOSJ8R/c9cOeNqup5SZ6XJIcccsgdKBEAAL7azIBcVXdN8pNJ7p/kI0ne2N3bV/natcJYr+W+3X1WkrOSZPPmzat9bVbgNrgAABPzplick2RzJuH4hCS/vAuvvS3JwcPyQUlu3A37AgDAHTZvisWR3f2wJKmqNyb5i1147UuTHFFVhyf56yQnJ3nmKve9MMnLhxPznhzXXQYAYDeYdwT5X3c82YWpFeP2p2USdq9J8rbuvqqqzqiqpyZJVR1TVduSPC3J66vqqum+Nyf5xUxC9qVJzpiOAQDAQs07gvyIqvr89Hkludt0uTK5RPI9Zu3c3RckuWCnsZcMzy/NZPrESvueneTsOfUBAMCamhmQu3vT7ioEAADWg9VeBxkAAPYKAjIAAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAYCMgAADARkAAAYCMgAADAQkAEAYCAgAwDAQEAGAICBgAwAAAMBGQAABgIyAAAMBGQAABgIyAAAMBCQAQBgICADAMBAQAYAgIGADAAAAwEZAAAGAjIAAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAYCMgAADARkAAAYCMgAADAQkAEAYCAgAwDAQEAGAIDBPssuAACS5OjTz112CbvFZWeesuwSgDkcQQYAgIGADAAAAwEZAAAGAjIAAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAYLDchVdXxVXVtV11XVi1dYf5eqeut0/Qer6rDp+GFVdWtVXTF9vG6RdQIAwA77LOqFq2pTktckeVKSbUkuraot3X31sNmPJ7mlu+9fVScneWWSH56u+0R3H7Wo+gAAYCWLPIJ8bJLruvv67v5SkvOSnLjTNicmOWf6/PwkT6iqWmBNAAAw0yID8oFJbhiWt03HVtymu7cn+Yck+0/XHV5VH6qq91XVv1vpC1TV86pqa1Vtvemmm9a2egAA9kqLDMgrHQnuVW7zmSSHdPcjk7wwyVuq6h5fs2H3Wd29ubs3H3DAAV93wQAAsMiAvC3JwcPyQUluvL1tqmqfJPsmubm7v9jdf5ck3X1Zkk8kecACawUAgCSLDciXJjmiqg6vqjsnOTnJlp222ZLk1Onzk5Jc3N1dVQdMT/JLVd03yRFJrl9grQAAkGSBV7Ho7u1VdVqSC5NsSnJ2d19VVWck2drdW5K8MclvVdV1SW7OJEQnyWOTnFFV25N8OclPdvfNi6oVAAB2WFhATpLuviDJBTuNvWR4/i9JnrbCfr+b5HcXWRsAAKzEnfQAAGAgIAMAwEBABgCAgYAMAAADARkAAAYCMgAADARkAAAYCMgAADAQkAEAYLDQO+kBALA+HH36ucsuYbe47MxTvu7XcAQZAAAGAjIAAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAYCMgAADARkAAAYCMgAADAQkAEAYCAgAwDAQEAGAICBgAwAAAMBGQAABgIyAAAMBGQAABgIyAAAMBCQAQBgICADAMBAQAYAgIGADAAAAwEZAAAGAjIAAAwEZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAYCMgAADARkAAAYCMgAADAQkAEAYCAgAwDAYJ9lF7C7HH36ucsuYbe47MxTll0CAMAezRFkAAAYCMgAADAQkAEAYLDQgFxVx1fVtVV1XVW9eIX1d6mqt07Xf7CqDhvW/dx0/Nqqesoi6wQAgB0WFpCralOS1yQ5IcmRSZ5RVUfutNmPJ7mlu++f5FeTvHK675FJTk7ykCTHJ3nt9PUAAGChFnkE+dgk13X39d39pSTnJTlxp21OTHLO9Pn5SZ5QVTUdP6+7v9jdf5XkuunrAQDAQlV3L+aFq05Kcnx3P2e6/KNJHtXdpw3bfHS6zbbp8ieSPCrJy5J8oLv/z3T8jUne3d3n7/Q1npfkedPFBya5diH/mTvuXkn+dtlFrGP6M5v+zKdHs+nPfHo0m/7Mpj/zrbceHdrdB8zbaJHXQa4VxnZO47e3zWr2TXefleSsXS9t96iqrd29edl1rFf6M5v+zKdHs+nPfHo0m/7Mpj/z7ak9WuQUi21JDh6WD0py4+1tU1X7JNk3yc2r3BcAANbcIgPypUmOqKrDq+rOmZx0t2WnbbYkOXX6/KQkF/dkzseWJCdPr3JxeJIjkvzFAmsFAIAkC5xi0d3bq+q0JBcm2ZTk7O6+qqrOSLK1u7ckeWOS36qq6zI5cnzydN+rquptSa5Osj3J87v7y4uqdYHW7fSPdUJ/ZtOf+fRoNv2ZT49m05/Z9Ge+PbJHCztJDwAA9kTupAcAAAMBGQAABgIyAAAMBGQAABgs8kYhe42qOiTJ33T3v0xvlf3sJN+RyVU43tDd25dZ33pQVU9N8sfd/S/LrmW9qqrHJvlcd19bVd+d5NFJrunudy25tHWjqu6e5PhMrpO+PcnHM/m++spSC1snqupBSU5McmAmN1e6McmW7r5mqYXBBlFVxybp7r60qo7M5OfRx7r7giWXtm5V1bndfcqy69hVrmKxBqa3zD62u79QVa9Mcr8k70jy+CTp7v+4zPrWg6q6Nck/J3l3kt9JcuEeeum+haiqX0tybCa/tF6Y5AmZ9Op7knyou09fYnnrQlU9PcnpST6c5Lgk78/kr2APS/Ks7v7IEstbuqp6UZJnJDkvk5stJZObLJ2c5LzufsWyalvvqurHuvtNy65jPZj+knVgkg929z8N48d39x8tr7Llq6qXJjkhk5/TFyV5VJJLkjwxk8+0X1pedetDVe18v4vK5Of1xUnS3U/d7UXdQQLyGqiqq7v7yOnzy5Ics+OIVlV9uLsfsdQC14Gq+lAmvzCclMkH9kOT/H6S3+nu9y2ztvWgqq7KpCd3S/LXSQ6c/sJ1p0wC8kOXWuA6UFVXJnn0tC/3SvLb3f2Uqnp4ktd193ctucSlqqq/TPKQ7v7XncbvnOSq7j5iOZWtf1X16e4+ZNl1LFtV/UyS5ye5JslRSV7Q3e+crru8u79jmfUtW1V9JJO+3CXJZ5Mc1N2fr6q7ZfILxcOXWuA6UFWXZ/LX89/M5K9YlclBsR33udhjPu9NsVgbN1TV47v74iSfzOTPv5+qqv2XW9a60t19S5I3JHlDVX1bkqcneUVVHdTdB8/efcPr7u6q2jFVYMdvrl+JcwV2qCS3Tp//c5JvTZLuvrKq7rG0qtaPryT59iSf2mn8PtN1e7XpL1grrkpy791Zyzr23CRHd/c/VdVhSc6vqsO6+1WZ9Glvt336l88vVNUnuvvzSdLdtw4/u/d2m5O8IMnPJzm9u6+oqlv3pGC8g4C8Np6T5NyqelmSf0hyxfSI6T2TvHCZha0jX/XDtbs/m+TVSV5dVYcup6R15V1V9f+S3DWT37zfVlUfyGSKxZ8stbL144Ikf1RV78vkz5xvT5Kq2i8+vJPkPyd5T1V9PMkN07FDktw/yWlLq2r9uHeSpyS5ZafxymS6DsmmHdMquvuTVfW4TELyofEeS5IvVdU3dvcXkhy9Y7Cq9o1fQpMk07+e/2pVvX367+eyh2ZNUyzWUFU9OMkDMvlm2JbkUicPTVTV47r7kmXXsZ5V1XdmciT5A1V1vyQ/mOTTSc73fTRRVd+b5MgkH+7ui6Zj35DkTt39xaUWtw5Me3FsJnNIK7f9HNrr5/tX1RuTvKm7/3SFdW/p7mcuoax1paouTvLC7r5iGNsnydmZzPPftLTi1oGqustKP2emU77us7efB7GSqvq+JI/p7v+27Fp2lYC8hqrq3hnOHu/uzy25pHVHj2bTn/n0aNdV1d3HE65gJVV1UCbTCD67wrrHdPefLaGsPYL32Hx7Wo8E5DVQVUcleV2SfTM5wSqZnD3+90l+ursvX1Zt64UezaY/8+nRHecktNn2tA/uZdCj2bzH5tvTerRHzgtZh96c5Ce6+4PjYFU9Osmbkuz1V7GIHs3z5ujPPG+OHt2uqrq98x0qyd13Zy17oKszma/N7dvre+Q9Nt9G6pGAvDa+aecP7SSZziX9pmUUtA7p0Wz6M58ezfbyJGdmcgOVne31V0LZSB/ci6JHc3mPzbdheiQgr413V9W7kpyb284ePzjJKUn26gurD/RoNv2ZT49muzzJO7r7sp1XVNVzllDPerNhPrgXSI9m8x6bb8P0yBzkNVJVJ+S2W7zuOHt8i9tP3kaPZtOf+fTo9lXVA5Pc3N03rbDu3nv7yYxV9f4k/+l2PrhvcC12PZrHe2y+jdQjARmADW8jfXAvih7BbfzJZA1U1b5V9Yqquqaq/m76uGY69i3Lrm890KPZ9Gc+PZpt6M/H9Odrdfe1KwW/6TrBL3o0j/fYfBupRwLy2nhbJndnOq679+/u/ZMcl8nlp96+1MrWDz2aTX/m06PZdvTncTv155boz4b64F4UPZrLe2y+DdMjUyzWQFVd290P3NV1exM9mk1/5tOj2fRntqq6MMnFSc7ZcSOMqvq2JKcmeWJ3P2mZ9a0HejSb99h8G6lHjiCvjU9V1c/W5A5fSSbztarqRbntbPu9nR7Npj/z6dFs+jPbYd39yvEucd392e5+Zfby6/sO9Gg277H5NkyPBOS18cNJ9k/yvqq6papuTnJJkv2SPH2Zha0jejSb/synR7Ppz2wb5oN7gfRoNu+x+TZMj0yxWCNV9aBMbnv7gfF2nFV1fHe7Rmv0aB79mU+PZtOf21dV90zy4kwuE3jvJJ3kc0m2JHlld9+8xPLWBT2az3tsvo3SI0eQ10BV/UySdyY5LclHq+rEYfXLl1PV+qJHs+nPfHo0m/7M1t23ZHJL8tOSHNzd+3X3g7v7RUmOXW5164MezeY9Nt9G6pE76a2N5yY5urv/qaoOS3J+VR3W3a/K5GYG6NE8+jOfHs2mPzNMP7ifn+SaJL9ZVS/o7ndOV7887saoR/N5j823YXokIK+NTTv+jNDdn6yqx2XyTXFo9rBviAXSo9n0Zz49mk1/ZtswH9wLpEezeY/Nt2F6ZIrF2vhsVR21Y2H6zfH9Se6V5GFLq2p90aPZ9Gc+PZpNf2b7qg/uJI9LckJV/Ur2sA/uBdKj2bzH5tswPXKS3hqoqoOSbB8vjTOse0x3/9kSylpX9Gg2/ZlPj2bTn9mq6uIkL+zuK4axfZKcneRZ3b1pacWtE3o0m/fYfBupRwIyABveRvrgXhQ9gtsIyAAAMDAHGQAABgIyAAAMBGSA3aiqvq2qzquqT1TV1VV1QVU9oKo+usCv+eyq+vXp85dV1V9X1RVV9fGq+r2qOnKVr/Hti6oRYD0RkAF2k6qqJL+f5JLuvl93H5nkv2VyW9/d6Ve7+6juPiLJW5NcXFUHzNnn2UkEZGCvICAD7D7HJfnX7n7djoHpJbVu2LFcVXetqjdV1Ueq6kNVddx0/CFV9RfTI79XVtUR0/EfGcZfX1WbpuM/VlV/WVXvS/KY2yuou9+a5I+TPHO630uq6tKq+mhVnVUTJyXZnOS3p1/nblV1dFW9r6ouq6oLq+o+a94tgCURkAF2n4cmuWzONs9Pku5+WJJnJDmnqu6a5CeTvKq7j8okrG6rqgcn+eEkj5mOfznJs6Zh9RcyCcZPSjJvCsXlSR40ff7r3X1Mdz80yd2SfH93n59kaybXwj0qyfYk/zvJSd19dCbXyf2l1TYBYL1zq2mA9eW7Mwmf6e6PVdWnkjwgyZ8n+fnptWp/r7s/XlVPSHJ0kksnszdytyR/k+RRmUzjuClJquqt09e4PeNd0o6rqp9N8o1J9ktyVZI/2Gn7B2YS9i+aft1NST5zh//HAOuMgAyw+1yV5KQ526x4S9/ufktVfTDJ9yW5sKqeM932nO7+ua96gaofSLIrF7l/ZJKt0yPVr02yubtvqKqXJbnr7dR4VXd/5y58DYA9hikWALvPxUnuUlXP3TFQVcckOXTY5k+SPGu67gFJDklybVXdN8n13f3qJFuSPDzJe5KcVFXfOt1+v6o6NMkHkzyuqvavqjsledrtFVRVP5TkyUl+J7eF4b+tqrvnq8P8Pyb55unza5McUFXfOX2NO1XVQ3a5GwDrlIAMsJv05NalP5jkSdPLvF2V5GVJbhw2e22STVX1kUyuMPHs7v5iJnONP1pVV2QyX/jc7r46yX9P8sdVdWWSi5Lcp7s/M33dP0/yfzOZYzz6Lzsu85bkR5I8vrtv6u6/T/KGJB9J8o4klw77vDnJ66Zff1Mm4fmVVfXhJFck+a6vu0EA64RbTQMAwMARZAAAGAjIAAAwEJABAGAgIAMAwEBABgCAgYAMAAADARkAAAb/H02mD4TKcrzNAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = default_year_close, x = 'ClosedDate', y= 'Proportion', color = base)\n",
    "plt.xticks(plt.xticks()[0], (default_year_close.ClosedDate.dt.year.astype(str)), rotation=90)\n",
    "plt.tight_layout();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [],
   "source": [
    "close = pd.DataFrame(df_loans.set_index('ClosedDate').groupby(pd.Grouper(freq='Y'))['ListingNumber'].count().reset_index())\n",
    "close['Proportion'] = close['ListingNumber']/total"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = close, x = 'ClosedDate', y= 'Proportion', color = base)\n",
    "plt.xticks(plt.xticks()[0], (close.ClosedDate.dt.year.astype(str)), rotation=90)\n",
    "plt.tight_layout();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> The loans that defaulted or got charged off have a higher percentage of 3Y contractual term, compared to the full population. \n",
    "> A super interesting insight came from looking at the creation and closed date of the loans: a great majority of the defaulted and charged off loans were closed in 2008 and 2009. \n",
    "\n",
    "### 9) Term vs Category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ListingNumber                    0\n",
       "ListingCreationDate              0\n",
       "CreditGrade                  83397\n",
       "Term                             0\n",
       "LoanStatus                       0\n",
       "ClosedDate                   57424\n",
       "ProsperRating (numeric)      28953\n",
       "ProsperScore                 28953\n",
       "ListingCategory (numeric)        0\n",
       "Occupation                    3568\n",
       "EmploymentStatus              2255\n",
       "IsBorrowerHomeowner              0\n",
       "Duration                     57424\n",
       "months                       57424\n",
       "Rating                           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "#store totals per Category and Term\n",
    "df_category = pd.DataFrame(df_loans.groupby(['ListingCategory (numeric)', 'Term'])['ListingNumber'].count()).reset_index()\n",
    "\n",
    "#store totals per Category\n",
    "category_aggregate = pd.DataFrame(df_category.groupby('ListingCategory (numeric)')['ListingNumber'].sum())\n",
    "\n",
    "#merge\n",
    "df_category = df_category.merge(category_aggregate, on = 'ListingCategory (numeric)', how = 'left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#get proportions\n",
    "df_category['Proportion'] = df_category['ListingNumber_x']/df_category['ListingNumber_y']\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = df_category, x = 'ListingCategory (numeric)', y = 'Proportion', hue = 'Term');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> For each of the category we have that the higher percentage comes from 3Y loans. For category 8 (Baby and Adoption), the proportion of 5Y loans get closer to the one of 3Y loans. Let's now look at the highest category for each term, ie we look at the relation between these two variables the other way round. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_12 = df_loans[df_loans.Term == 12]\n",
    "df_36 = df_loans[df_loans.Term == 36]\n",
    "df_60 = df_loans[df_loans.Term == 60]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "order_12 = df_12['ListingCategory (numeric)'].value_counts().index.tolist()\n",
    "total_12 = df_12.shape[0]\n",
    "\n",
    "ax_12 = sb.countplot(data = df_12, x = 'ListingCategory (numeric)', color = base, order = order_12)\n",
    "\n",
    "for p in ax_12.patches:\n",
    "    height = p.get_height()\n",
    "    ax_12.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_12),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "order_36 = df_36['ListingCategory (numeric)'].value_counts().index.tolist()\n",
    "total_36 = df_36.shape[0]\n",
    "\n",
    "ax_36 = sb.countplot(data = df_36, x = 'ListingCategory (numeric)', color = base, order = order_36)\n",
    "\n",
    "for p in ax_36.patches:\n",
    "    height = p.get_height()\n",
    "    ax_36.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_36),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAnQAAAFACAYAAAA1auHpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3X2c1XWd///HSxANTbnU0MFQQRI0L8AL1rKSFLMWtNXCTeWb9vO3pV3YN1PX/ablutpW63ZtliT0VUezEttVlPWicksRvAZLSEwBUwq00hUDX98/zmfwgGdgmJlzznyYx/12m9uc8/68P+f1fp/hzDx5f87nfCIzkSRJUnlt1ewBSJIkqWsMdJIkSSVnoJMkSSo5A50kSVLJGegkSZJKzkAnSZJUcgY6SZKkkjPQSZIklZyBTpIkqeT6NnsAjTZkyJAcMWJEs4chSZK0SfPnz/9DZg7dVL9eF+hGjBjBvHnzmj0MSZKkTYqI33Wkn4dcJUmSSs5AJ0mSVHIGOkmSpJIz0EmSJJWcgQ6YPXs2o0ePZuTIkVx66aWv237WWWex//77s//++7PXXnsxYMCAddtmzJjBqFGjGDVqFDNmzGjksCVJkgCIzGz2GBpq/PjxWX2W69q1a9lrr72YM2cOLS0tHHTQQVx77bWMGTOm5v5f//rXeeCBB5g+fTorV65k/PjxzJs3j4hg3LhxzJ8/n4EDBzZqOpIkaQsWEfMzc/ym+vX6Fbq5c+cycuRI9thjD/r168fUqVOZNWtWu/2vvfZaTjzxRABuvfVWjjzySAYNGsTAgQM58sgjmT17dqOGLkmSBBjoWLZsGcOHD193v6WlhWXLltXs+7vf/Y4lS5ZwxBFHbPa+kiRJ9dLrA12tQ84RUbNva2srxx9/PH369NnsfSVJkuql1we6lpYWnn766XX3ly5dyi677FKzb2tr67rDrZu7ryRJUr30+kB30EEHsWjRIpYsWcIrr7xCa2srkydPfl2/3/zmN6xatYoJEyasa5s0aRK33XYbq1atYtWqVdx2221MmjSpkcOXJEnqfddyrTbu7JkAbHXAcew9bgL56qsM3vdwTrlqPsvvPp/+bxrBgJEHArD8v39C7rwP4z/7g/Ue49XRExm2x1sAeNMhf8uRl/wHAPO/dEoDZyJJknqzXh3o2uy4x37suMd+67Xt8rb3r3//sONq7jtk38MZsu/hdRubJEnSpvT6Q66SJEllZ6CTJEkqOQOdJElSyRnoJEmSSs5AJ0mSVHIGOkmSpJIz0EmSJJWcgU6SJKnkDHSSJEklV7dAFxHTI+K5iHh0g/aPR8RvImJBRPxrVft5EbG42Dapqv3oom1xRJxb1b57RNwbEYsi4rqI6FevuUiSJPVk9Vyhuwo4urohIt4FTAHempljgS8X7WOAqcDYYp9vRUSfiOgDfBN4DzAGOLHoC/BF4LLMHAWsAk6r41wkSZJ6rLoFusz8ObByg+aPApdm5uqiz3NF+xSgNTNXZ+YSYDFwcPG1ODOfyMxXgFZgSkQEcARwQ7H/DODYes1FkiSpJ2v0e+j2At5eHCr9WUQcVLTvCjxd1W9p0dZe+2Dg+cxcs0F7TRFxekTMi4h5K1as6KapSJIk9QyNDnR9gYHAocDZwPXFalvU6JudaK8pM6/IzPGZOX7o0KGbP2pJkqQerG+D6y0FfpyZCcyNiFeBIUX78Kp+LcDy4nat9j8AAyKib7FKV91fkiSpV2n0Ct2NVN77RkTsBfSjEs5uAqZGxDYRsTswCpgL3AeMKs5o7UflxImbikB4J3B88bjTgFkNnYkkSVIPUbcVuoi4FngnMCQilgIXANOB6cVHmbwCTCvC2YKIuB5YCKwBzsjMtcXjnAncCvQBpmfmgqLEOUBrRPwz8ABwZb3mIkmS1JPVLdBl5ontbDqpnf4XAxfXaL8ZuLlG+xNUzoKVJEnq1bxShCRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5OoW6CJiekQ8FxGP1tj2mYjIiBhS3I+I+FpELI6IhyPiwKq+0yJiUfE1rap9XEQ8UuzztYiIes1FkiSpJ6vnCt1VwNEbNkbEcOBI4Kmq5vcAo4qv04FvF30HARcAhwAHAxdExMBin28Xfdv2e10tSZKk3qBugS4zfw6srLHpMuCzQFa1TQFmZsU9wICIGAZMAuZk5srMXAXMAY4utu2Qmb/KzARmAsfWay6SJEk9WUPfQxcRk4FlmfnQBpt2BZ6uur+0aNtY+9Ia7e3VPT0i5kXEvBUrVnRhBpIkST1PwwJdRPQHzgc+V2tzjbbsRHtNmXlFZo7PzPFDhw7tyHAlSZJKo5ErdHsCuwMPRcSTQAtwf0S8icoK2/Cqvi3A8k20t9RolyRJ6nUaFugy85HM3CkzR2TmCCqh7MDM/D1wE3BKcbbrocALmfkMcCtwVEQMLE6GOAq4tdj254g4tDi79RRgVqPmIkmS1JPU82NLrgV+BYyOiKURcdpGut8MPAEsBr4LfAwgM1cCFwH3FV9fKNoAPgp8r9jnt8At9ZiHJElST9e3Xg+cmSduYvuIqtsJnNFOv+nA9Brt84B9ujZKSZKk8vNKEZIkSSVnoJMkSSo5A50kSVLJGegkSZJKzkAnSZJUcgY6SZKkkjPQSZIklZyBTpIkqeQMdJIkSSVnoJMkSSo5A50kSVLJGegkSZJKzkAnSZJUcgY6SZKkkjPQSZIklZyBTpIkqeQMdJIkSSVnoJMkSSo5A50kSVLJGegkSZJKzkAnSZJUcnULdBExPSKei4hHq9q+FBG/joiHI+InETGgatt5EbE4In4TEZOq2o8u2hZHxLlV7btHxL0RsSgirouIfvWaiyRJUk9WzxW6q4CjN2ibA+yTmW8FHgfOA4iIMcBUYGyxz7ciok9E9AG+CbwHGAOcWPQF+CJwWWaOAlYBp9VxLpIkST1W3QJdZv4cWLlB222Zuaa4ew/QUtyeArRm5urMXAIsBg4uvhZn5hOZ+QrQCkyJiACOAG4o9p8BHFuvuUiSJPVkzXwP3anALcXtXYGnq7YtLdraax8MPF8VDtvaa4qI0yNiXkTMW7FiRTcNX5IkqWdoSqCLiPOBNcDVbU01umUn2mvKzCsyc3xmjh86dOjmDleSJKlH69voghExDXgfMDEz20LYUmB4VbcWYHlxu1b7H4ABEdG3WKWr7i9JktSrNHSFLiKOBs4BJmfmS1WbbgKmRsQ2EbE7MAqYC9wHjCrOaO1H5cSJm4ogeCdwfLH/NGBWo+YhSZLUk9TzY0uuBX4FjI6IpRFxGvAN4I3AnIh4MCIuB8jMBcD1wEJgNnBGZq4tVt/OBG4FHgOuL/pCJRh+OiIWU3lP3ZX1moskSVJPVrdDrpl5Yo3mdkNXZl4MXFyj/Wbg5hrtT1A5C1aSJKlX80oRkiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSq1ugi4jpEfFcRDxa1TYoIuZExKLi+8CiPSLiaxGxOCIejogDq/aZVvRfFBHTqtrHRcQjxT5fi4io11wkSZJ6snqu0F0FHL1B27nA7Zk5Cri9uA/wHmBU8XU68G2oBEDgAuAQ4GDggrYQWPQ5vWq/DWtJkiT1CnULdJn5c2DlBs1TgBnF7RnAsVXtM7PiHmBARAwDJgFzMnNlZq4C5gBHF9t2yMxfZWYCM6seS5IkqVdp9Hvods7MZwCK7zsV7bsCT1f1W1q0bax9aY32miLi9IiYFxHzVqxY0eVJSJIk9SQ95aSIWu9/y06015SZV2Tm+MwcP3To0E4OUZIkqWdqdKB7tjhcSvH9uaJ9KTC8ql8LsHwT7S012iVJknqdRge6m4C2M1WnAbOq2k8pznY9FHihOCR7K3BURAwsToY4Cri12PbniDi0OLv1lKrHkiRJ6lX61uuBI+Ja4J3AkIhYSuVs1UuB6yPiNOAp4ISi+83AMcBi4CXgwwCZuTIiLgLuK/p9ITPbTrT4KJUzad8A3FJ8SZIk9Tp1C3SZeWI7mybW6JvAGe08znRgeo32ecA+XRmjJEnSlqBDh1wj4vaOtEmSJKnxNrpCFxHbAv2pHDYdyGtnl+4A7FLnsUmSJKkDNnXI9f8HPkUlvM3ntUD3J+CbdRyXJEmSOmijgS4zvwp8NSI+nplfb9CYJEmStBk6dFJEZn49Iv4GGFG9T2bOrNO4JEmS1EEdCnQR8QNgT+BBYG3R3HYNVUmSJDVRRz+2ZDwwpvh4EUmSJPUgHb1SxKPAm+o5EEmSJHVOR1fohgALI2IusLqtMTMn12VUkiRJ6rCOBroL6zkISZIkdV5Hz3L9Wb0HIkmSpM7p6Fmuf6ZyVitAP2Br4MXM3KFeA5MkSVLHdHSF7o3V9yPiWODguoxIkiRJm6WjZ7muJzNvBI7o5rFIkiSpEzp6yPX9VXe3ovK5dH4mnSRJUg/Q0bNc/7bq9hrgSWBKt49GkiRJm62j76H7cL0HIkmSpM7p0HvoIqIlIn4SEc9FxLMR8aOIaKn34CRJkrRpHT0p4vvATcAuwK7AT4s2SZIkNVlHA93QzPx+Zq4pvq4ChtZxXJIkSeqgjga6P0TESRHRp/g6CfhjPQcmSZKkjulooDsV+ADwe+AZ4Hig0ydKRMRZEbEgIh6NiGsjYtuI2D0i7o2IRRFxXUT0K/puU9xfXGwfUfU45xXtv4mISZ0djyRJUpl1NNBdBEzLzKGZuROVgHdhZwpGxK7AJ4DxmbkP0AeYCnwRuCwzRwGrgNOKXU4DVmXmSOCyoh8RMabYbyxwNPCtiOjTmTFJkiSVWUcD3Vszc1XbncxcCRzQhbp9gTdERF+gP5VVvyOAG4rtM4Bji9tTivsU2ydGRBTtrZm5OjOXAIvxcmSSJKkX6mig2yoiBrbdiYhBdPxDideTmcuALwNPUQlyLwDzgeczc03RbSmVs2kpvj9d7Lum6D+4ur3GPpIkSb1GR0PZV4BfRsQNVC759QHg4s4ULILhFGB34Hngh8B7anRtu7RYtLOtvfZaNU8HTgfYbbfdNnPEkiRJPVuHVugycybwd8CzwArg/Zn5g07WfDewJDNXZOZfgR8DfwMMKA7BArQAy4vbS4HhAMX2HYGV1e019tlw/Fdk5vjMHD90qJ+2IkmStiwdPeRKZi7MzG9k5tczc2EXaj4FHBoR/Yv3wk0EFgJ3Ujl7FmAaMKu4fVNxn2L7HZmZRfvU4izY3YFRwNwujEuSJKmUOvU+uK7IzHuLQ7f3A2uAB4ArgP8EWiPin4u2K4tdrgR+EBGLqazMTS0eZ0FEXE8lDK4BzsjMtQ2djCRJUg/Q8EAHkJkXABds0PwENc5SzcyXgRPaeZyL6eR7+SRJkrYUHT7kKkmSpJ7JQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJdeUQBcRAyLihoj4dUQ8FhETImJQRMyJiEXF94FF34iIr0XE4oh4OCIOrHqcaUX/RRExrRlzkSRJarZmrdB9FZidmW8B9gMeA84Fbs/MUcDtxX2A9wCjiq/TgW8DRMQg4ALgEOBg4IK2EChJktSbNDzQRcQOwOHAlQCZ+UpmPg9MAWYU3WYAxxa3pwAzs+IeYEBEDAMmAXMyc2VmrgLmAEc3cCqSJEk9QjNW6PYAVgDfj4gHIuJ7EbEdsHNmPgNQfN+p6L8r8HTV/kuLtvbaXyciTo+IeRExb8WKFd07G0mSpCZrRqDrCxwIfDszDwBe5LXDq7VEjbbcSPvrGzOvyMzxmTl+6NChmzteSZKkHq0ZgW4psDQz7y3u30Al4D1bHEql+P5cVf/hVfu3AMs30i5JktSrNDzQZebvgacjYnTRNBFYCNwEtJ2pOg2YVdy+CTilONv1UOCF4pDsrcBRETGwOBniqKJNkiSpV+nbpLofB66OiH7AE8CHqYTL6yPiNOAp4ISi783AMcBi4KWiL5m5MiIuAu4r+n0hM1c2bgqSJEk9Q1MCXWY+CIyvsWlijb4JnNHO40wHpnfv6CRJksrFK0VIkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJWegkyRJKjkDnSRJUsk1LdBFRJ+IeCAi/qO4v3tE3BsRiyLiuojoV7RvU9xfXGwfUfUY5xXtv4mISc2ZiSRJUnM1c4Xuk8BjVfe/CFyWmaOAVcBpRftpwKrMHAlcVvQjIsYAU4GxwNHAtyKiT4PGLkmS1GM0JdBFRAvwXuB7xf0AjgBuKLrMAI4tbk8p7lNsn1j0nwK0ZubqzFwCLAYObswMJEmSeo5mrdD9O/BZ4NXi/mDg+cxcU9xfCuxa3N4VeBqg2P5C0X9de419JEmSeo2GB7qIeB/wXGbOr26u0TU3sW1j+2xY8/SImBcR81asWLFZ45UkSerpmrFCdxgwOSKeBFqpHGr9d2BARPQt+rQAy4vbS4HhAMX2HYGV1e019llPZl6RmeMzc/zQoUO7dzaSJElN1vBAl5nnZWZLZo6gclLDHZn5IeBO4Pii2zRgVnH7puI+xfY7MjOL9qnFWbC7A6OAuQ2ahiRJUo/Rd9NdGuYcoDUi/hl4ALiyaL8S+EFELKayMjcVIDMXRMT1wEJgDXBGZq5t/LAlSZKaq6mBLjPvAu4qbj9BjbNUM/Nl4IR29r8YuLh+I5QkSer5vFKEJElSyRnoJEmSSs5AJ0mSVHIGOkmSpJIz0EmSJJWcgU6SJKnkDHSSJEklZ6CTJEkqOQOdJElSyRnoJEmSSs5AJ0mSVHIGOkmSpJIz0EmSJJWcgU6SJKnkDHSSJEklZ6CTJEkqOQOdJElSyRnoJEmSSs5AJ0mSVHIGOkmSpJIz0EmSJJWcga5JZs+ezejRoxk5ciSXXnrp67avXr2aD37wg4wcOZJDDjmEJ598EoC//vWvTJs2jX333Ze9996bSy65pMEjlyRJPU3DA11EDI+IOyPisYhYEBGfLNoHRcSciFhUfB9YtEdEfC0iFkfEwxFxYNVjTSv6L4qIaY2eS2etXbuWM844g1tuuYWFCxdy7bXXsnDhwvX6XHnllQwcOJDFixdz1llncc455wDwwx/+kNWrV/PII48wf/58vvOd76wLe5IkqXdqxgrdGuB/Z+bewKHAGRExBjgXuD0zRwG3F/cB3gOMKr5OB74NlQAIXAAcAhwMXNAWAnu6uXPnMnLkSPbYYw/69evH1KlTmTVr1np9Zs2axbRplYx6/PHHc/vtt5OZRAQvvvgia9as4X/+53/o168fO+ywQzOmIUmSeoiGB7rMfCYz7y9u/xl4DNgVmALMKLrNAI4tbk8BZmbFPcCAiBgGTALmZObKzFwFzAGObuBUOm3ZsmUMHz583f2WlhaWLVvWbp++ffuy44478sc//pHjjz+e7bbbjmHDhrHbbrvxmc98hkGDBjV0/JIkqWdp6nvoImIEcABwL7BzZj4DldAH7FR02xV4umq3pUVbe+216pweEfMiYt6KFSu6cwqdkpmva4uIDvWZO3cuffr0Yfny5SxZsoSvfOUrPPHEE3UbqyRJ6vmaFugiYnvgR8CnMvNPG+taoy030v76xswrMnN8Zo4fOnTo5g+2m7W0tPD0069l0aVLl7LLLru022fNmjW88MILDBo0iGuuuYajjz6arbfemp122onDDjuMefPmdbh2Z0/GAHj44YeZMGECY8eOZd999+Xll1/ezJlLkqR6aEqgi4itqYS5qzPzx0Xzs8WhVIrvzxXtS4HhVbu3AMs30t7jHXTQQSxatIglS5bwyiuv0NrayuTJk9frM3nyZGbMqByBvuGGGzjiiCOICHbbbTfuuOMOMpMXX3yRe+65h7e85S0dqtuVkzHWrFnDSSedxOWXX86CBQu466672Hrrrbvh2ZAkSV3Vt9EFo3Js8Urgscz8t6pNNwHTgEuL77Oq2s+MiFYqJ0C8kJnPRMStwL9UnQhxFHBeI+bQFePOngnAVgccx97jJpCvvsrgfQ/nlKvms/zu8+n/phEMGHkgr67ZhifvvJ+rBu5Mn223Y/f3fYxxZ89k7SsD+d0vF9A6tAUSBu/zdj78gweBBwGY/6VT2q1dfTIGsO5kjDFjxqzrM2vWLC688EKgcjLGmWeeSWZy22238da3vpX99tsPgMGDB9fh2ZEkSZ3R8EAHHAacDDwSEQ8Wbf9IJchdHxGnAU8BJxTbbgaOARYDLwEfBsjMlRFxEXBf0e8LmbmyMVPouh332I8d99hvvbZd3vb+dbe36tuPPSaf+br9+vTbtmZ7R9Q6GePee+9tt0/1yRiPP/44EcGkSZNYsWIFU6dO5bOf/WyH6s6ePZtPfvKTrF27lo985COce+65621fvXo1p5xyCvPnz2fw4MFcd911jBgxgieffJK9996b0aNHA3DooYdy+eWXd2rukiRtyRoe6DLzbmq//w1gYo3+CZzRzmNNB6Z33+i2bF05GWPNmjXcfffd3HffffTv35+JEycybtw4Jk583Y9sPW2HeefMmUNLSwsHHXQQkydPXm9VsPowb2trK+eccw7XXXcdAHvuuScPPvhgew8vSZLwShG9SldOxmhpaeEd73gHQ4YMoX///hxzzDHcf//9m6zZlc/ckyRJHWOg60W6cjLGpEmTePjhh3nppZdYs2YNP/vZz9ZbZWtPVz5zD2DJkiUccMABvOMd7+AXv/hFl+YvSdKWqhnvoVODtZ2IAZ0/GQNg5bCDGLzbKCDYYY/9+Nxdf+Rzd83c6IkYXTnMO2zYMJ566ikGDx7M/PnzOfbYY1mwYIFXxpAkaQMGul6msydjAAwecxiDxxy2WfU25zBvS0vLeod5I4JtttkGgHHjxrHnnnvy+OOPM378+M0agyRJWzoPuaquunKYd8WKFaxduxaAJ554gkWLFq37yBVJkvQaV+hUN139zL1Vj9/HM//9Y2KrPhBbMeywD3DkJf8BbPzz9iRJ6m0MdKq7zh7mHbjXQQzc66C6j0+SpLLzkKskSVLJGegkSZJKzkAnSZJUcgY6bZFmz57N6NGjGTlyJJdeeunrtq9evZoPfvCDjBw5kkMOOYQnn3wSgDlz5jBu3Dj23Xdfxo0bxx133NHgkUuStPkMdNritF0/9pZbbmHhwoVce+21LFy4cL0+1dePPeusszjnnHMAGDJkCD/96U955JFHmDFjBieffPJm1TZISpKawUCnLU5Xrh97wAEHrPvg47Fjx/Lyyy+zevXqDtVtVpA0REqSDHTa4nT1+rFtfvSjH3HAAQesu1rFpjQjSJZxNfKPf/wj73rXu9h+++0588zaVyXpSTUlqQwMdNridOX6sW0WLFjAOeecw3e+850O121GkCzjauS2227LRRddxJe//OUO1WpmTTBESioHA522OJtz/VhgvevHtvU/7rjjmDlzJnvuuWeH6zYjSJZxNXK77bbjbW97G9tuu22HajWzZrNCJBgkJW0eA522OF25fuzzzz/Pe9/7Xi655BIOO+ywzarbjCBZ9tXIzdGMms0IkVC+1UiASy65hJEjRzJ69GhuvfXWHl1T2hIZ6LRFGXf2TA4575p11499404trNhhFKdcNZ9hE45lz+M+xbizZ/K9xdtw3Z33s+3AnTntrH9k4Y4HM+7smYx532k8uvDXnPzRT9N/pzfTf6c389aPfYNxZ89cd23a9jQjSJZ5NXJzNaNmM0IklG81cuHChbS2trJgwQJmz57Nxz72MdauXdsja7bpTeG1t8y1Nz2/tXgtV22ROnv92GETpjBswpTNqlUd9NqCZL76KoP3PZxTrprP8rvPp/+bRjBg5IG8umYbnrzzfq4auDN9tt2O3d/3McadPZNnfjWLZ4sgefJHPw3AyOPPZuvtdmD+l05pt3Z1iNx1111pbW3lmmuuWa9PW4icMGFCU1YjW1paXhckO6MZNZsRIqF2kLz33nvb7VMdJIcMGdKpmtUhElgXIseMGbOuz6xZs7jwwguBSog888wzyUxmzZrF1KlT2Wabbdh9990ZOXIkc+fOZcKECT2uJrwWJOfMmUNLSwsHHXQQkydPXq9udZBsbW3lnHPO4brrrlsvSC5fvpx3v/vdPP744/Tp06fH1exNc+1Nz297XKGTutGOe+zH2NP+lX3+vy8z7NDK6twub3s/A0YeCLwWJMd+5Eu85aQL2WbATkAlSO7/qe+y97SL1n1tvd0OG61Vz9XITenKamRnNaNmV1dAO6tsq5Ed2ben1ISurYC2FyR7Ys3eNNfe9Py2xxU6qeTKthoJ8OgV/5u1r/wPuXYNV8y4hpHHn80bhuy60dXIvn378o1vfINJkyaxdu1aTj31VMaOHcvnPvc5xo8fz+TJkznttNM4+eSTGTlyJIMGDaK1tXXd/iNGjOBPf/oTr7zyCjfeeCO33Xbbev+LrqUrK6BdUbbVyM6Gy2bUhK6tgC5btoxDDz10vX07G17rXbM3zbU3Pb/tKX2gi4ijga8CfYDvZebrD2BL6nadDZIA+5z+lc2qVR0k3zjlnwD48fPw47NnAiP5z188z+d/UfQZ8bfsOOJvWQuc8O27gbsBGHzCFxhc9Zgnf38eMK/dENlWs9HBFZoTJLsGaGCwAAAOfElEQVQSIjuyb0+pCb0rvPaWufam57c9pQ50EdEH+CZwJLAUuC8ibsrMhRvfU5I6plnBtdFBsishcvLkyfz93/89n/70p1m+fDmLFi3i4IMP3uR8m1ETeld47S1z7U3Pb3tKHeiAg4HFmfkEQES0AlMAA52kUmtGkOxKiFyxwyh22Hk4sVUfWt719xx87tUAdV0Bba/mxupC7wqvvWWuven5bU/ZA92uwNNV95cChzRpLJJUal0JkcMOnbzuRKCeWrO7VkB7UnjdVM3eMFfo2ntsx44dywc+8AHGjBlD3759+eY3v9nhM02bVbeWqHUMtywi4gRgUmZ+pLh/MnBwZn58g36nA6cXd0cDv+lEuSHAH7ow3M5qRt3eUrNZdZ3rllezWXV7S81m1XWuW17NZtXtSs03Z+bQTXUq+wrdUmB41f0WYPmGnTLzCuCKrhSKiHmZOb4rj1GWur2lZrPqOtctr2az6vaWms2q61y3vJrNqtuImmX/HLr7gFERsXtE9AOmAjc1eUySJEkNVeoVusxcExFnArdS+diS6Zm5oMnDkiRJaqhSBzqAzLwZuLkBpbp0yLZkdXtLzWbVda5bXs1m1e0tNZtV17lueTWbVbfuNUt9UoQkSZLK/x46SZKkXs9AJ0mSVHIGuk2IiOkR8VxEPNrAmsMj4s6IeCwiFkTEJxtUd3REPFj19aeI+FSda24bEXMj4qFirp+vY63X/Swj4qKIeLiY720R0fnrrnS85oURsazqeT6mO2vWGMOAiLghIn5d/JuaUKc67b5WIuIzEZERMaSba9Z8rUTEoIiYExGLiu8Du7lurZ/r/hFxT/EznRcRnf/I947X3C8ifhURj0TETyNih3rXLNo/HhG/KZ7zf+3OmjXGcFZR59GIuDYitq1TnVrP7wlF7Vcjou4fcxERnyzmuaCev3vbmWszXjPXVf0efDIiHmxAzS8VvwsfjoifRMSA7qzZzjiOLl4viyPi3LoVyky/NvIFHA4cCDzawJrDgAOL228EHgfGNHjefYDfU/lAw3rWCWD74vbWwL3AoY36WQI7VN3+BHB5A2peCHymgT/LGcBHitv9gAGNen6L9uFUzkT/HTCkm2vWfK0A/wqcW7SfC3yxAT/X24D3FLePAe5qQM37gHcUt08FLmpAzXcB/wVsU9zfqR7/norH3hVYAryhuH898L/qVKvWXPem8mH0dwHj6zXPotY+wKNAfyonLP4XMKqBc234a2aD7V8BPteAeR4F9C1uf7G751ljDH2A3wJ7FL9/H6JOf89doduEzPw5sLLBNZ/JzPuL238GHqPyi62RJgK/zczf1bNIVvyluLt18VWXM3Vq/Swz809Vd7fr7trN+PdTrVixORy4shjPK5n5fD1qbWSulwGfpQ4/1428VqZQCbIU34/t5rq15ppA2wrZjtT4kPM61BwN/Ly4PQf4uwbU/ChwaWauLvo81501a+gLvCEi+lIJO936vLZp5/fDY5nZmSsLdcbewD2Z+VJmrgF+BhxXj0Lt/Fyb8ZoBICIC+ABwbb1rZuZtxfMLcA+VCxLU07przmfmK0DbNee7nYGuh4uIEcABVFauGmkq3fziak9E9CmW2p8D5mRmQ+caERdHxNPAh4DPNajsmcWS//TuPrSxgT2AFcD3I+KBiPheRGxXx3rriYjJwLLMfKgBtUbw2mtl58x8BiqhD9ip3vWBTwFfKv4tfRk4rwE1HwXaLmZ6AutfOade9gLeHhH3RsTPIuKgehXKzGVUnsungGeAFzLztnrVa7JHgcMjYnBE9KeyytuIn2ebZrxm2rwdeDYzFzWwJlRWtW+pc41a15yvywKNga4Hi4jtgR8Bn9pgJanedftR+SPxw0bUy8y1mbk/lf8pHRwR+zSiblX98zNzOHA1UPsq4N3r28CewP5U/kh9pY61+lI55PDtzDwAeJHK4ZS6K/4onU8DQnKzXisb+ChwVvFv6SyKVdE6OxU4IyLmUznk/EoDavYFBgKHAmcD1xcrLN2u+M/OFGB3YBdgu4g4qR61mi0zH6NyCHAOMJvKobk1G91py3EiDVpAaBMR51N5fq+ud6kabXU5CmWg66EiYmsqf6CuzswfN7j8e4D7M/PZRhYtDgXeBRzdyLpVrqGbD1nVkpnPFiH2VeC7VJbk62UpsLRq1fMGKgGvEfak8of4oYh4kkpgvz8i3tSdRdp5rTwbEcOK7cOorP7W2zSgrf4Pqe/PFYDM/HVmHpWZ46j8QfxtvWtS+Tf14+LtEnOBV6lceLwe3g0sycwVmflXKs/v39SpVtNl5pWZeWBmHk7lUGEjV6ya8ZqhOJT+fuC6RtQrak4D3gd8KIs3utVRh6453x0MdD1Q8b/dK4HHMvPfmjCEhv1vKSKGtp1lFBFvoPIL/NeNqF3UHFV1d3Ijarf90iwcR+VQS11k5u+BpyNidNE0EVhYr3ob1H4kM3fKzBGZOYLKL7YDizF1i428Vm6iErAovs/qrpobsRx4R3H7CBrwxzgidiq+bwX8E3B5vWsCN1KZHxGxF5U3ev+hTrWeAg6NiP7Fz3oilfdJbpGqfp67UQk5jVy1asZrBorf+Zm5tBHFIuJo4Bxgcma+1ICSjbvmfD3OtNiSvqi8oJ4B/krlD9JpDaj5NipLsg8DDxZfxzRovv2BPwI7NqjeW4EHirk+Sjef5bSpnyWVlZ1Hi/o/BXZtQM0fAI8UNW8ChtX5Od4fmFfUuxEY2Kjnd4PtT9L9Z7nWfK0Ag4HbqYSq24FBDfi5vg2YT+VQ2b3AuAbU/CSVM3sfBy6luPpPnWv2A/5v8bq5Hziizv9+P0/lP1qPFq+dbRr175fKf7iWAquBZ4Fb6zzXX1D5D9dDwMQ61qk114a/Zor2q4B/aOA8F1N5T1vb74tu/WSDdsZxTPEa/S1wfr3qeOkvSZKkkvOQqyRJUskZ6CRJkkrOQCdJklRyBjpJkqSSM9BJkiSVnIFOUreIiL/UaPuHiDhlI/u8MyL+pqP9OzCG7SPiOxHx24hYEBE/j4hDNrHPP3a2Xj1ExLER0ahL0G1SRNzc9lmR7Ww/MyI+3MgxSXo9P7ZEUreIiL9k5vabuc+FwF8y88vdNIZWYAmVz3p6NSL2APbOzP/cyD6bPe5OjKtvvnZB8E31/SWVDz2t14f1dkjxQb6RlSuabKxff+C/s3JpOUlN4gqdpLqJiAsj4jPF7U9ExMKIeDgiWiNiBPAPwFkR8WBEvH2D/ndFxBcjYm5EPB4Rby/a+0fE9cXjXFdcJH58ROwJHAL8U1sIycwn2sJcRNwYEfOLlbvTi7ZLgTcU9a8u2k4qaj5YrPb1KdpPK8ZxV0R8NyK+UbS/OSJuL8Zze/Ep/0TEVRHxbxFxJ/CliFgUEUOLbVtFxOKIWO+SWcWVF1a3hbniMb4WEb+MiCci4vii/Z0R8R9V+30jIv5XcfvJiPiXiPhVRMyLiAMj4tZi1fIfqvY5OyLuK8b9+aJtREQ8FhHfovKhwcOLxxtSbD+l6P9QRPygeI5fAp6MiLpf6kxS+/o2ewCSeo1zgd0zc3VEDMjM5yPicqpW6CJi4gb79M3MgyPiGOACKpcJ+hiwKjPfGhH7UPm0d4CxwIOZubad+qdm5sqoXGLuvoj4UWaeGxFnZub+Rf29gQ8Ch2XmX4tg86GI+C/g/1C5Du6fgTuofJo/wDeAmZk5IyJOBb4GHFts2wt4d2aujYjngQ8B/17M46Eaq3CHUQlS1YZRuQrFW6hcWeSGduZX7enMnBARl1H5JP7DgG2BBcDlEXEUMIrK9WYDuCkiDqdyqa3RwIcz82PFc0LxfSxwfvHc/CEiBlXVmwe8HZjbgbFJqgNX6CQ1ysPA1RFxEtChw4+8drH7+cCI4vbbgFaAzGy7bFtHfCIiHgLuoXKx7FE1+kwExlEJfA8W9/egEnx+lpkrs3KR+B9W7TMBuKa4/YNifG1+WBUwpwNt7w88Ffh+jfrDgBUbtN2Yma9m5kJg501PE3jtWpGPAPdm5p8zcwXwcvF+uKOKrweoBMi38Nrz8bvMvKfGYx4B3NAWQjNzZdW254BdOjg2SXXgCp2kRnkvcDgwGfg/xYrPpqwuvq/ltd9X0U7fBcB+EbHVhu/7ioh3UlkVm5CZL0XEXVRWrDYUwIzMPG+D/Y/rwFjbVL8x+cV1jZlPR8SzEXEElUPDH6qx7/8AO27Qtrrqdtvc17D+f8g3nEvbPq9usP+rVJ7HAC7JzO9U71QcBn+R2oL151Zt22LskprEFTpJdRcRWwHDM/NO4LPAAGB7Kocv37iZD3c38IHicccA+wJk5m+pHPr7fPGGfiJiVERMoRKSVhVh7i3AoVWP99eI2Lq4fTtwfETsVOw/KCLeTOVQ4jsiYmBE9AX+rmr/XwJTi9sfKsbXnu9RubD99e0cGn4MGNmB5+B3wJiI2CYidqSykrg5bgVOjYjtASJi17Y5b8TtwAciYnCxT/Uh172ARzdzDJK6kYFOUnfpHxFLq74+XbWtD/B/I+IRKof5LsvM54GfAscVJyC8vYN1vgUMjYiHgXOoHHJ9odj2EeBNwOKi1neB5cBsoG+xz0VUDru2uQJ4OCKuLg5r/hNwW9F3DjAsM5cB/wLcC/wXsLCq5ieADxf9TwY+uZGx30QlyNY63Arwc+CAtkDansx8Gri+mPvVVJ7TDsvM26gcJv5V8TzdwCaCdWYuAC4GflYcuv63qs2HUXleJDWJH1siqVSKs063zsyXo3Jm6+3AXpn5Sp3rbp+ZfylW6H4CTM/Mn2zmY4ynEmbbDa8R8VXgp5lZioAUEQcAn87Mk5s9Fqk38z10ksqmP3BncZg0gI/WO8wVLoyId1N5v9htwI2bs3NEnAt8lNrvnav2L1TeY1cWQ6icASypiVyhkyRJKjnfQydJklRyBjpJkqSSM9BJkiSVnIFOkiSp5Ax0kiRJJff/AGYGlE4soMiZAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "order_60 = df_60['ListingCategory (numeric)'].value_counts().index.tolist()\n",
    "total_60 = df_60.shape[0]\n",
    "\n",
    "ax_60 = sb.countplot(data = df_60, x = 'ListingCategory (numeric)', color = base, order = order_60)\n",
    "\n",
    "for p in ax_60.patches:\n",
    "    height = p.get_height()\n",
    "    ax_60.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_60),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Here we note that higher the term, higher is the proportion of loans for Debt Consolidation.\n",
    "\n",
    "### 10) Term vs Rating\n",
    "> Do borrowers with higher credit rating get a loan with higher term?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_term_rating = pd.DataFrame(df_loans.groupby(['Rating', 'Term']).count()['ListingNumber']).reset_index()\n",
    "df_term_rating['Proportion'] = df_term_rating['ListingNumber']/112350"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmsAAAFACAYAAADjzzuMAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3XuYXXV97/H3l1wxQaIhUXCI4W5jIxFC4DQUlQAlPZbIESHAqUECVCxo9YGqpxRD1ALtY7FFKofrE7UYLopGTOEUIqhRgRAuCTcJSGEKCoSbsSRkku/5Y6/BzWQm2aGzZq2Zeb+eZx7WWvu39/7MhcxnfusWmYkkSZLqaZuqA0iSJKlnljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqjHLmiRJUo1Z1iRJkmrMsiZJklRjQ6sO0Ft22GGHnDhxYtUxJEmStuiuu+56LjPHtTJ2wJS1iRMnsmzZsqpjSJIkbVFE/EerY90NKkmSVGOWNUmSpBqzrEmSJNXYgDlmrTvr16+nvb2dtWvXVh3lDRk5ciRtbW0MGzas6iiSJKkiA7qstbe3s9122zFx4kQiouo4WyUzWb16Ne3t7eyyyy5Vx5EkSRUZ0LtB165dy9ixY/tdUQOICMaOHdtvZwUlSVLvGNBlDeiXRa1Tf84uSZJ6x4Ava5IkSf3ZgD5mbWutXr2aGTNmAPDrX/+aIUOGMG5c4+LCd9xxB8OHD68yniRJGoQsa03Gjh3LPffcA8C8efMYPXo0Z5xxRsvP37BhA0OGDCkrniRJGoTcDdqiBQsWMG3aNKZMmcInPvEJNm7cSEdHB2PGjOGss85i2rRp3HHHHbS1tfE3f/M3HHDAAey3334sX76cww47jN12241LL7206k9DkiT1M86stWDlypVcf/31/OxnP2Po0KGccsopLFy4kKOPPpqXXnqJffbZhy996UuvjZ84cSK/+MUvOP3005k7dy4//elPWbNmDXvvvTcnn3xyhZ+JBqsn5k9uadyEs1eUnESStLUsay24+eabufPOO5k6dSoAr7zyCjvvvDMAw4cP58gjj3zd+COOOAKAyZMn09HRwahRoxg1ahTbbLMNa9asYfTo0X37CUiSpH7LstaCzOTEE0/ki1/84uu2d3R0sO22225yiY0RI0YAsM0227y23Lne0dFRfmBJkjRgeMxaCw455BCuueYannvuOaBx1ugTTzxRcSpJkjQYOLPWgsmTJ/OFL3yBQw45hI0bNzJs2DAuvvhidtppp6qjDXqtHIvlcViSpP7MstaDefPmvW79uOOO47jjjttk3Isvvvi69fb29teWTzrppB4fkyRJaoW7QSVJkmrMsiZJklRjljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNTaoLt2x75nf6NXXu+sfPrrFMSeeeCI33HAD48ePZ+XKlQCceeaZ/OAHP2D48OHstttuXHnllYwZM6ZXs0mSpIFhUJW1KpxwwgmcdtppfPSjvy92hx56KOeeey5Dhw7ls5/9LOeeey7nn39+hSm3zBuBS5JUDXeDluyggw7irW996+u2HXbYYQwd2ujJBxxwgBfLlSRJPbKsVeyKK65g5syZVceQJEk1ZVmr0Je//GWGDh3K8ccfX3UUSZJUUx6zVpEFCxZwww03cMsttxARVceRJEk1ZVmrwI033sj555/Pbbfdxpve9Kaq40iSpBobVGWtlUtt9LZjjz2WW2+9leeee462tjbOOecczj33XNatW8ehhx4KNE4yuPjii/s8myRJqr9BVdaq8O1vf3uTbXPnzq0giSRJ6o88wUCSJKnGLGuSJEk1ZlmTJEmqMcuaJElSjVnWJEmSasyyJkmSVGOD6tIdT8yf3KuvN+HsFVscs3btWg466CDWrVtHR0cHRx11FOeccw6ZyVlnncW1117LkCFDOPXUU/nkJz/Zq/kkSVL/N6jKWhVGjBjBkiVLGD16NOvXr+fAAw9k5syZPPjggzz55JM89NBDbLPNNjzzzDNVR5UkSTVkWStZRDB69GgA1q9fz/r164kIvv71r3PVVVexzTaNPdHjx4+vMqYkSaopj1nrAxs2bGDKlCmMHz+eQw89lP33359HH32Uq6++mqlTpzJz5kweeeSRqmNKkqQaKrWsRcThEfFwRKyKiM918/iIiLi6ePz2iJhYbB8WEQsiYkVEPBgRny8zZ9mGDBnCPffcQ3t7O3fccQcrV65k3bp1jBw5kmXLlnHyySdz4oknVh1TkiTVUGllLSKGABcBM4FJwLERManLsLnAC5m5O3ABcH6x/SPAiMycDOwL/EVnkevPxowZw/vf/35uvPFG2tra+PCHPwzAkUceyX333VdxOkmSVEdlzqxNA1Zl5mOZ+SqwEJjVZcwsYEGxfB0wIyICSGBURAwFtgVeBV4uMWtpnn32WV588UUAXnnlFW6++Wbe9a538aEPfYglS5YAcNttt7HnnntWGVOSJNVUmScYvAN4smm9Hdi/pzGZ2RERLwFjaRS3WcDTwJuAT2fm813fICJOAU4BmDBhwhYDtXKpjd729NNPM2fOHDZs2MDGjRs5+uij+eAHP8iBBx7I8ccfzwUXXMDo0aO57LLL+jybytHqJWKq+HmUJPU/ZZa16GZbtjhmGrAB2Al4C/CTiLg5Mx973cDMS4BLAKZOndr1tWvhPe95D3ffffcm28eMGcMPf/jDChJJkqT+pMzdoO3Azk3rbcBTPY0pdnluDzwPHAfcmJnrM/MZYCkwtcSskiRJtVRmWbsT2CMidomI4cBsYFGXMYuAOcXyUcCSzEzgCeDgaBgFHAA8VGJWSZKkWiqtrGVmB3AacBPwIHBNZt4fEfMj4ohi2OXA2IhYBXwG6Ly8x0XAaGAljdJ3ZWZ6uqQkSRp0Sr2DQWYuBhZ32XZ20/JaGpfp6Pq8Nd1tlyRJGmy8g4EkSVKNWdYkSZJqbFDdyH36hdN79fWWnr60pXEvvvgiJ510EitXriQiuOKKK9hrr7045phjePzxx5k4cSLXXHMNb3nLW3o1nyRJ6v+cWesDn/rUpzj88MN56KGHuPfee/mDP/gDzjvvPGbMmMEjjzzCjBkzOO+886qOKUmSasiyVrKXX36ZH//4x8ydOxeA4cOHM2bMGL7//e8zZ07jqiVz5szhe9/7XpUxJUlSTVnWSvbYY48xbtw4Pvaxj/He976Xk046id/97nf85je/YccddwRgxx135Jlnnqk4qSRJqiPLWsk6OjpYvnw5p556KnfffTejRo1yl6ckSWqZZa1kbW1ttLW1sf/+jXvYH3XUUSxfvpy3ve1tPP3000DjZu/jx4+vMqYkSaopy1rJ3v72t7Pzzjvz8MMPA3DLLbcwadIkjjjiCBYsWADAggULmDVrVpUxJUlSTQ2qS3e0eqmN3nbhhRdy/PHH8+qrr7Lrrrty5ZVXsnHjRo4++mguv/xyJkyYwLXXXltJNkmSVG+DqqxVZcqUKSxbtmyT7bfccksFaSRJUn/iblBJkqQas6xJkiTV2IAva5lZdYQ3rD9nlyRJvWNAl7WRI0eyevXqfll6MpPVq1czcuTIqqNIkqQKDegTDNra2mhvb+fZZ5+tOsobMnLkSNra2qqOIUmSKjSgy9qwYcPYZZddqo4hSZL0hg3o3aCSJEn9nWVNkiSpxixrkiRJNWZZkyRJqjHLmiRJUo0N6LNBJYDpF05vadzS05eWnESSpK3nzJokSVKNWdYkSZJqzLImSZJUY5Y1SZKkGrOsSZIk1Zhng9bQE/MntzRuwtkrSk4iSZKq5syaJElSjVnWJEmSasyyJkmSVGOWNUmSpBqzrEmSJNWYZU2SJKnGLGuSJEk1ZlmTJEmqMcuaJElSjVnWJEmSasyyJkmSVGOWNUmSpBqzrEmSJNWYZU2SJKnGLGuSJEk1ZlmTJEmqsaFVB6jaE/MntzRuwtkrSk4iSZK0KWfWJEmSaqzUshYRh0fEwxGxKiI+183jIyLi6uLx2yNiYtNj74mIn0fE/RGxIiJGlplVkiSpjkoraxExBLgImAlMAo6NiEldhs0FXsjM3YELgPOL5w4FvgV8PDPfDbwfWF9WVkmSpLoqc2ZtGrAqMx/LzFeBhcCsLmNmAQuK5euAGRERwGHAfZl5L0Bmrs7MDSVmlSRJqqUyy9o7gCeb1tuLbd2OycwO4CVgLLAnkBFxU0Qsj4i/7u4NIuKUiFgWEcueffbZXv8EJEmSqlZmWYtutmWLY4YCBwLHF/89MiJmbDIw85LMnJqZU8eNG/ffzStJklQ7ZZa1dmDnpvU24KmexhTHqW0PPF9svy0zn8vM/wIWA/uUmFWSJKmWyixrdwJ7RMQuETEcmA0s6jJmETCnWD4KWJKZCdwEvCci3lSUuPcBD5SYVZIkqZZKuyhuZnZExGk0itcQ4IrMvD8i5gPLMnMRcDnwzYhYRWNGbXbx3Bci4h9pFL4EFmfmD8vKKkmSVFel3sEgMxfT2IXZvO3spuW1wEd6eO63aFy+Q5IkadDyDgaSJEk1ZlmTJEmqMcuaJElSjVnWJEmSaqylEwwiYk/gTOCdzc/JzINLyiVJkiRaPxv0WuBi4FLAe3RKkiT1kVbLWkdmfr3UJJIkSdpEq8es/SAiPhERO0bEWzs/Sk0mSZKklmfWOm8JdWbTtgR27d04kiRJatZSWcvMXcoOIkmSpE21ejboMOBU4KBi063A/83M9SXlkga86RdOb2nc0tOXlpxEklRnre4G/TowDPiXYv3Pi20nlRFKkiRJDa2Wtf0yc++m9SURcW8ZgSRJkvR7rZ4NuiEidutciYhd8XprkiRJpWt1Zu1M4EcR8RgQNO5k8LHSUkmSJAlo/WzQWyJiD2AvGmXtocxcV2oySZIkbb6sRcTBmbkkIv5Xl4d2iwgy87slZpMkSRr0tjSz9j5gCfBn3TyWgGVNkiSpRJsta5n5hWJxfmb+qvmxiPBCuZIkSSVr9WzQ73Sz7breDCJJkqRNbemYtXcB7wa273Lc2puBkWUGkyRJ0paPWdsL+CAwhtcft/Zb4OSyQkmStLWemD+5pXETzl5RchKpd23pmLXvR8QNwGcz8+/6KJMkSZIKWzxmLTM3AIf2QRZJkiR10eodDH4WEV8DrgZ+17kxM5eXkkqSJElA62Xtj4r/zm/alsDBvRtHkiRJzVq93dQHyg4iSZKkTbV0nbWI2D4i/jEilhUfX4mI7csOJ0mSNNi1elHcK2hcruPo4uNl4MqyQkmSJKmh1WPWdsvMDzetnxMR95QRSJIkSb/X6szaKxFxYOdKREwHXiknkiRJkjq1OrN2KrCgOE4tgOeBOaWlkiRJEtD62aD3AHtHxJuL9ZdLTSVJkiSg9bNBx0bEPwO3Aj+KiH+KiLGlJpMkSVLLu0EXAj8GOk8yOJ7G3QwOKSOUJEkDgTeXV29otay9NTO/2LT+pYj4UBmBJEmS9Hutng36o4iYHRHbFB9HAz8sM5gkSZJaL2t/AVwFvFp8LAQ+ExG/jQhPNpAkSSpJq2eDbld2EEmSJG2q1WPWiIgjgIOK1Vsz84ZyIkmSJKlTq5fuOA/4FPBA8fGpYpskSZJK1OrM2p8CUzJzI0BELADuBj5XVjBJkiS1foIBwJim5e17O4gkSZI21erM2rnA3RHxIxr3Bj0I+HxpqSRJkgS0UNYiIoCfAgcA+9Eoa5/NzF+XnE2SJGnQ22JZy8yMiO9l5r7Aoj7IJEmSpEKrx6z9IiL2KzWJJEmSNtFqWfsAjcL2aETcFxErIuK+LT0pIg6PiIcjYlVEbHLmaESMiIiri8dvj4iJXR6fEBFrIuKMFnNKkiQNKK2eYDBza184IoYAFwGHAu3AnRGxKDMfaBo2F3ghM3ePiNnA+cAxTY9fAPzb1r63JEnSQLHZshYRI4GPA7sDK4DLM7OjxdeeBqzKzMeK11oIzKJxUd1Os4B5xfJ1wNciIorj5D4EPAb8rsX3kyRJGnC2tBt0ATCVRlGbCXxlK177HcCTTevtxbZuxxQl8CVgbESMAj4LnLO5N4iIUyJiWUQse/bZZ7cimiRJUv+wpd2gkzJzMkBEXA7csRWvHd1syxbHnANckJlrGlcO6V5mXgJcAjB16tSury1JktTvbamsre9cyMyOzRWnbrQDOzettwFP9TCmPSKG0rgzwvPA/sBREfH3NO6csDEi1mbm17YmgCRJUn+3pbK2d0S8XCwHsG2xHjQuwfbmzTz3TmCPiNgF+E9gNnBclzGLgDnAz4GjgCWZmcAfdw6IiHnAGouaJNXHE/MntzRuwtkrSk4iDXybLWuZOeSNvnAxE3cacBMwBLgiM++PiPnAssxcBFwOfDMiVtGYUZv9Rt9PkiRpIGr10h1vSGYuBhZ32XZ20/Ja4CNbeI15pYSTJEnqB1q9KK4kSZIqYFmTJEmqsVJ3g2rwmX7h9JbGLT19aclJJEkaGJxZkyRJqjHLmiRJUo1Z1iRJkmrMsiZJklRjljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqjHvYCCpEk/Mn9zSuAlnryg5iSTVmzNrkiRJNWZZkyRJqjHLmiRJUo1Z1iRJkmrMsiZJklRjljVJkqQa89IdklRzrVzmxEucqL/z57xnzqxJkiTVmDNrkl4z/cLpLY1bevrSkpNIkjo5syZJklRjljVJkqQaczeoJBW8X6mkOrKsSZI0iPhHSf/jblBJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqjHLmiRJUo1Z1iRJkmrMsiZJklRjljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqjHLmiRJUo1Z1iRJkmrMsiZJklRjljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqrFSy1pEHB4RD0fEqoj4XDePj4iIq4vHb4+IicX2QyPirohYUfz34DJzSpIk1VVpZS0ihgAXATOBScCxETGpy7C5wAuZuTtwAXB+sf054M8yczIwB/hmWTklSZLqrMyZtWnAqsx8LDNfBRYCs7qMmQUsKJavA2ZERGTm3Zn5VLH9fmBkRIwoMaskSVItlVnW3gE82bTeXmzrdkxmdgAvAWO7jPkwcHdmrisppyRJUm0NLfG1o5ttuTVjIuLdNHaNHtbtG0ScApwCMGHChDeWUpIkqcbKnFlrB3ZuWm8DnuppTEQMBbYHni/W24DrgY9m5qPdvUFmXpKZUzNz6rhx43o5viRJUvXKLGt3AntExC4RMRyYDSzqMmYRjRMIAI4ClmRmRsQY4IfA5zNzaYkZJUmSaq20slYcg3YacBPwIHBNZt4fEfMj4ohi2OXA2IhYBXwG6Ly8x2nA7sDfRsQ9xcf4srJKkiTVVZnHrJGZi4HFXbad3bS8FvhIN8/7EvClMrNJkiT1B97BQJIkqcYsa5IkSTVmWZMkSaqxUo9ZU7mmXzi9pXFLT/eEWkmS+itn1iRJkmrMsiZJklRjljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqjHLmiRJUo1Z1iRJkmrMsiZJklRjljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqjHLmiRJUo1Z1iRJkmrMsiZJklRjljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqjHLmiRJUo1Z1iRJkmrMsiZJklRjljVJkqQas6xJkiTVmGVNkiSpxixrkiRJNWZZkyRJqrGhVQeQpM2ZfuH0lsYtPX1pyUkkqRrOrEmSJNWYZU2SJKnGLGuSJEk1ZlmTJEmqMcuaJElSjVnWJEmSasyyJkmSVGNeZ22Q2/fMb7Q07vrtSg7SD/i1kiRVwbLWh/xl3zq/Vq3x6yRJA5+7QSVJkmrMmTVJvcrZPknqXaWWtYg4HPgnYAhwWWae1+XxEcA3gH2B1cAxmfl48djngbnABuCTmXlTmVklSZKaPTF/ckvjJpy9otQcpe0GjYghwEXATGAScGxETOoybC7wQmbuDlwAnF88dxIwG3g3cDjwL8XrSZIkDSplHrM2DViVmY9l5qvAQmBWlzGzgAXF8nXAjIiIYvvCzFyXmb8CVhWvJ0mSNKiUWdbeATzZtN5ebOt2TGZ2AC8BY1t8riRJ0oAXmVnOC0d8BPiTzDypWP9zYFpmnt405v5iTHux/iiNGbT5wM8z81vF9suBxZn5nS7vcQpwSrG6F/BwL8XfAXiul16rt5ipdXXMZabWmKl1dcxlptaYqXV1zNVbmd6ZmeNaGVjmCQbtwM5N623AUz2MaY+IocD2wPMtPpfMvAS4pBczAxARyzJzam+/7n+HmVpXx1xmao2ZWlfHXGZqjZlaV8dcVWQqczfoncAeEbFLRAynccLAoi5jFgFziuWjgCXZmOpbBMyOiBERsQuwB3BHiVklSZJqqbSZtczsiIjTgJtoXLrjisy8PyLmA8sycxFwOfDNiFhFY0ZtdvHc+yPiGuABoAP4y8zcUFZWSZKkuir1OmuZuRhY3GXb2U3La4GP9PDcLwNfLjPfZvT6rtVeYKbW1TGXmVpjptbVMZeZWmOm1tUxV59nKu0EA0mSJP33eW9QSZKkGrOsSZIk1digLWsRcUVEPBMRK3t4PCLinyNiVUTcFxH79EGmnSPiRxHxYETcHxGfqjpXRIyMiDsi4t4i0zndjBkREVcXmW6PiIllZmp63yERcXdE3FCjTI9HxIqIuCcilnXzeBU/V2Mi4rqIeKj42fofVWaKiL2Kr0/nx8sR8VdVZtpM1iMjIiPiXVW8/5ZyRMSnI2JtRGxfYbYNxffx3ohYHhF/VFWWZhHx9ohYGBGPRsQDEbE4IvasME/n1+n+4mv1mYio/HdwU67Oj89VkGFNl/UTIuJrxfK8iPjPItsDEXFsH+bKiPhK0/oZETGvaf2jEbGy+J4+EBFnlBYmMwflB3AQsA+wsofH/xT4NyCAA4Db+yDTjsA+xfJ2wC+BSVXmKt5ndLE8DLgdOKDLmE8AFxfLs4Gr++h7+BngKuCGbh6rKtPjwA6bebyKn6sFwEnF8nBgTNWZmt57CPBrGheHrEWmLjmuAX4CzKvi/beUg8YljX4CnFBhtjVNy38C3Fbl16rIEcDPgY83bZsC/HFNvk7jgZuBc2rwtVpTtwzACcDXiuV5wBnF8h7Ay8CwPsq1FvhV57/pwBmd/w/SuO/5cmCnYn0kcHJZWSpv9VXJzB/TuFxIT2YB38iGXwBjImLHkjM9nZnLi+XfAg+y6W22+jRX8T6df/UMKz66npXS0z1eSxMRbcD/BC7rYUifZ2pRn37/IuLNNP4wuRwgM1/NzBerzNTFDODRzPyPGmUCICJGA9OBuRSXFapCTzkiYjdgNHAW0GezDVvwZuCFqkMAHwDWZ+bFnRsy857M/EmFmV6Tmc/QuPvOaTX5d6lfyMxHgP8C3tJHb9lB48zPT3fz2OdplMinimxrM/PSsoIM2rLWgkrvT1rstnsvjZmsZn2eq9jdeA/wDPDvmdljpnz9PV7L9FXgr4GNPTxeRSZoFNn/FxF3ReN2aD3mKpT9/dsVeBa4sthlfFlEjKo4U7PZwLe72V6H+wN/CLgxM38JPF/VrtjN5DiWxtfuJ8BeETG+onzbFruoHqLxx9MXK8rR7A+Bu6oOsTmZ+RiN38FVfd86dX7/Oj+OqToDjVtObqL42X+kKLt95SLg+G4ONejTnzHLWs+6+2unT65zUvwl/R3grzLz5a4Pd/OUUnNl5obMnELjtl/TIuIPq8wUER8EnsnMzf2PUtX3b3pm7kNjivwvI+KgLo/3da6hNHb3fz0z3wv8Duh6TEolX6to3NnkCODa7h7uZltfX2foWGBhsbyQ6mavesoxG1iYmRuB79LDNSv7wCuZOSUz3wUcDnzD2aKW1eHr1Pn96/y4uuoMwNldHv90RDxMY/JiXl8GK34HfwP4ZF++b1eWtZ61dH/S3hYRw2gUtX/NzO/WJRdAsfvsVhr/IHebKV5/j9eyTAeOiIjHafzyOjgivlVxJgCapsSfAa4HpvWUq1D2968daG+aDb2ORnmrMlOnmcDyzPxNN49V9nMOEBFjgYOBy4qfszOBY/q6hGwmx940jt/592L7bGqwKzQzf07jJtct3Zy6RPcD+1acYbMiYldgA409Ftq8CzJzL+AYGn8MjOzj9/8qjcMQmvdK9OnPmGWtZ4uAjxZnpR0AvJSZT5f5hsUvgsuBBzPzH+uQKyLGRcSYYnlb4BDgoW4ydXeP11Jk5uczsy0zJ9L4JbUkM/93lZkAImJURGzXuQwcBnQ927hPv3+Z+WvgyYjYq9g0g8Zt3CrL1KRzN153qsrU6Sgax8y9MzMnZubONA40PrAPM2wux1dpHOg8sfjYCXhHRLyzj/O9TjTOVh0CrK4yB7AEGBERJ3duiIj9IuJ9FWZ6TUSMAy6mcRC9V6ZvUTGBsYzf/9veV+/7PI2TfOY2bT4X+PuIeDu8dgWC0mbfSr3dVJ1FxLeB9wM7REQ78AUaB89THJS6mMYZaatoHND4sT6INR34c2BFsd8e4P8AEyrMtSOwICKG0Cj312TmDdHCPV77Wg0yvQ24vph8GQpclZk3RsTHodKfq9OBfy12Oz4GfKzqTBHxJuBQ4C+atlX9dWp2LHBel23fAY6jcYxY1Tk+TWPmttn1NH7Oz++DXM22bfr3KoA5WfG9nDMzI+JI4KvRuBTFWhpnav/VZp9Yrs6v0zAaB65/E+jpj/K+1Pz9g8bxkX1++Y6tMB+4KiIuLQ4B6CtfAU7rXMnMxRHxNuDmYqIlgSvKenNvNyVJklRj7gaVJEmqMcuaJElSjVnWJEmSasyyJkmSVGOWNUmSpBqzrEka8CJiQ3Erm5UR8YPOawduZvyYiPhE0/pOEXFd+UklaVNeukPSgBcRazJzdLG8APhlZn55M+MnAjdkZtdbq0lSn3NmTdJg83OKm8JHxOiIuCUilkfEioiYVYw5D9itmI37h4iYGBEri+ecEBHfjYgbI+KRiPj7zheOiLkR8cuIuDUiLo2Ir/X5ZydpwBm0dzCQNPgUd+KYQeMOF9C4sv2RmflyROwA/CIiFtG42f0fFjeV7pxpazYFeC+wDng4Ii5mJtbcAAABPUlEQVSkcZ/Hv6Vx79Xf0rjl0b2lfkKSBgXLmqTBoPOWOhOBu4B/L7YH8HcRcRCwkcaM29taeL1bMvMlgIh4AHgnjRuY31bcR5CIuBbYszc/CUmDk7tBJQ0GrxSzZO8EhgN/WWw/HhgH7Fs8/htgZAuvt65peQONP3yj9+JK0u9Z1iQNGsVs2CeBMyJiGLA98Exmro+ID9Aoc9DYjbndVr78HcD7IuItETEU+HBv5ZY0uFnWJA0qmXk3jWPJZgP/CkyNiGU0ZtkeKsasBpYWl/r4hxZf9z+BvwNuB24GHgBe6v3PQNJg46U7JKmXRMTozFxTzKxdD1yRmddXnUtS/+bMmiT1nnnFiQwrgV8B36s4j6QBwJk1SZKkGnNmTZIkqcYsa5IkSTVmWZMkSaoxy5okSVKNWdYkSZJq7P8DH05xDB6i6BEAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "sb.barplot(data = df_term_rating, x = 'Rating', y = 'Proportion', hue = 'Term');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Only 3Y loans seem to be present before 2009, so let's focus on the period after 2009 and at the Prosper score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "order_12 = df_12['ProsperScore'].value_counts().index.tolist()\n",
    "total_12 = df_12.ProsperScore.value_counts().sum()\n",
    "\n",
    "ax_12 = sb.countplot(data = df_12, x = 'ProsperScore', color = base, order = order_12)\n",
    "\n",
    "for p in ax_12.patches:\n",
    "    height = p.get_height()\n",
    "    ax_12.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_12),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6.61"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "0.19*8+0.18*6+10*0.12+0.12*7+5*0.1+9*0.1+4*0.08+2*0.05+3*0.04+1*0.03"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "order_36 = df_36['ProsperScore'].value_counts().index.tolist()\n",
    "total_36 = df_36.ProsperScore.value_counts().sum()\n",
    "\n",
    "ax_36 = sb.countplot(data = df_36, x = 'ProsperScore', color = base, order = order_36)\n",
    "\n",
    "for p in ax_36.patches:\n",
    "    height = p.get_height()\n",
    "    ax_36.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_36),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.88"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "4*0.15+8*0.14+0.14*6+0.12*5+0.12*7+0.1*3+0.09*9+2*0.08 +10*0.06 + 0.01*1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "order_60 = df_60['ProsperScore'].value_counts().index.tolist()\n",
    "total_60 = df_60.ProsperScore.value_counts().sum()\n",
    "\n",
    "ax_60 = sb.countplot(data = df_60, x = 'ProsperScore', color = base, order = order_60)\n",
    "\n",
    "for p in ax_60.patches:\n",
    "    height = p.get_height()\n",
    "    ax_60.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_60),\n",
    "            ha=\"center\") \n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.949999999999999"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "6*0.17+4*0.15+0.15*7+0.15*8+0.12*5+3*0.08+9*0.07+10*0.05+2*0.05+1*0.01"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Unfortunately, I cannot derive any insights from here as it seems that 1Y loans have a higher rating but 5Y loans have higher rating than 3Y loans. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Talk about some of the relationships you observed in this part of the investigation. How did the feature(s) of interest vary with other features in the dataset?\n",
    "\n",
    "> We considered credit rating as one of the main features in the analysis. This varied in accordance to common sense and expectations. Some of the insights below: \n",
    "   - Prosper Score and Prosper Rating are highly correlated variables.\n",
    "   - People with a lower Prosper score or Credit Rating tend to have a higher chance to have their loans charge off or default. This also tells us that the rating is a good and useful indicator for Prosper, which is always good news!\n",
    "   -  There is a peak of homeowners where rating is AA/A or 6/7 and the amount is decreasing with lower credit ratings. \n",
    "   - There is a smaller proportion of homeowners in the sample of defaulted and charged off loans, compared to the full population.\n",
    "\n",
    "### Did you observe any interesting relationships between the other features (not the main feature(s) of interest)?\n",
    "\n",
    "> One other feature of interest was the duration of the loans. **Before 2009, only 3Y loans were offered**, which obviously makes the analysis by loan term more complex. Also, a lot of defaulted/charged off loans were closed in 2009. \n",
    "> \n",
    "> Among the completed loans, ie excluding current and past due, generally speaking **1Y and 3Y loans lasted the whole term**, whereas **5Y loans were completed earlier**. \n",
    ">\n",
    "> However, this might due to the fact that the first 5Y loan was created in 2010 and the dataset is until 2014 so we would **need data for the following years** as well to make conclusions on whether 5Y loans get repaid earlier or not. \n",
    ">\n",
    "> Another interesting insight is that, even though the most common loan categories remain the same across the different loan terms, higher the term, **higher is the proportion of loans for Debt Consolidation**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Multivariate Exploration\n",
    "\n",
    "> Here we look at Loan Status, mainly Defaulted and Charged Off loans, term and duration (ie closed date - creation date).\n",
    "\n",
    "### Difference between Term and Duration\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_loans['TimeDiff'] = df_loans['Term']- df_loans['months']\n",
    "\n",
    "def new_column(x):\n",
    "    if x<-12:\n",
    "        return \"Closed >1Y after term date\"\n",
    "    if x <0 and x>=-12:\n",
    "        return \"Closed <1Y after term date\"\n",
    "    elif x>=0 and x<=12:\n",
    "        return \"Closed <1Y before term date\"\n",
    "    elif x>12 and x<25:\n",
    "        return \"Closed 1Y-2Y before term date\"\n",
    "    elif x>24 and x<37:\n",
    "        return \"Closed 2Y-3Y before term date\"\n",
    "    elif x>36 and x<49:\n",
    "        return \"Closed 3Y-4Y before term date\"\n",
    "    elif x>48 and x<61:\n",
    "        return \"Closed 4Y-5Y before term date\"\n",
    "    \n",
    "df_loans['TimeDiff_Cat'] = df_loans['TimeDiff'].apply(lambda t:new_column(t))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "plt.hist(data = df_loans, x = 'TimeDiff');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "default_off_cat = df_loans[df_loans['LoanStatus'].isin(['Chargedoff', 'Defaulted'])]\n",
    "total_default_off = default_off_cat[default_off_cat['TimeDiff_Cat'].notnull()]['ListingNumber'].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "plt.hist(data=default_off_cat, x = 'TimeDiff'); "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> While loans in general seem to be closed according to the original term, charged off and defaulted loans are generally closed earlier, but let's take a closer look. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "total = df_loans[df_loans['TimeDiff_Cat'].notnull()]['ListingNumber'].count()\n",
    "order_time = [\"Closed >1Y after term date\", 'Closed <1Y after term date', 'Closed <1Y before term date', 'Closed 1Y-2Y before term date', \"Closed 2Y-3Y before term date\", \"Closed 3Y-4Y before term date\", \"Closed 4Y-5Y before term date\"]\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "time = sb.countplot(data = df_loans, x = 'TimeDiff_Cat', color = base, order = order_time)\n",
    "plt.xticks(rotation=90)\n",
    "\n",
    "for p in time.patches:\n",
    "    height = p.get_height()\n",
    "    time.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total),\n",
    "            ha=\"center\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAm4AAAHQCAYAAAAYgOaLAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzs3XucVXW9//HXG0YwNREUTR08SMMhwSR1EKwsLwnaqbE6aHRRTtrRTtjF+nkp85aZdOxYmeU5FibaydHsAqcUJPJaR3AQRB0zKFBATQS8pEeQ8fP7Y62BGRmYYWb2XvPd834+HvOYvb7ru/d89vfx3TOfWet7UURgZmZmZj1fn6IDMDMzM7OOceJmZmZmlggnbmZmZmaJcOJmZmZmlggnbmZmZmaJcOJmZmZmlggnbmZmZmaJcOJmZmZmlggnbmZmZmaJqCrVC0saAdzcomgYcCFwQ14+FFgOnBQR6yQJ+B7wfuAV4F8i4sH8tSYDX8tf5xsRMX1bP3uPPfaIoUOHdtt7MTMzMyuVBQsWPBcRgztSV+XY8kpSX2AVMBaYAqyNiKmSzgMGRsS5kt4PfI4scRsLfC8ixkoaBDQAtUAAC4BDI2Ld1n5ebW1tNDQ0lPZNmZmZmXUDSQsiorYjdct1q/QY4C8R8QRwAtB8xWw68KH88QnADZG5H9hN0t7ABGBORKzNk7U5wHFlitvMzMysxyhX4jYJuCl/vFdEPA2Qf98zL98XWNHiOSvzsq2Vm5mZmfUqJU/cJPUD6oCft1e1jbLYRvkbf87pkhokNaxevXr7AzUzMzPr4cpxxe144MGI+Ft+/Lf8Fij592fz8pXAkBbPqwae2kZ5KxFxbUTURkTt4MEdGt9nZmZmlpRyJG4fY/NtUoCZwOT88WRgRovyU5QZB7yQ30qdDYyXNFDSQGB8XmZm1m1mzZrFiBEjqKmpYerUqVutd+uttyKJlhOgLr/8cmpqahgxYgSzZ/vXk5mVTsmWAwGQtBNwLHBGi+KpwC2STgOeBE7My28jm1G6lGw5kE8BRMRaSZcCD+T1vh4Ra0sZt5n1Lk1NTUyZMoU5c+ZQXV3NmDFjqKurY+TIka3qvfTSS1x11VWMHTt2U1ljYyP19fU8+uijPPXUU7zvfe/jz3/+M3379i332zCzXqCkV9wi4pWI2D0iXmhRtiYijomI4fn3tXl5RMSUiHhrRLw9IhpaPOe6iKjJv35SypjNrPeZP38+NTU1DBs2jH79+jFp0iRmzJixRb0LLriAc845hx133HFT2YwZM5g0aRL9+/dn//33p6amhvnz55czfDPrRbxzgpn1eqtWrWLIkM1Daaurq1m1alWrOgsXLmTFihV84AMf2O7nmpl1l5LeKjUzS0FbC5Fnm7lkXn/9dc466yyuv/767X6umVl3cuJmZr1edXU1K1ZsXi5y5cqV7LPPPpuOX3rpJR555BGOPPJIAJ555hnq6uqYOXNmu881M+tOvlVqZr3emDFjWLJkCcuWLWPDhg3U19dTV1e36fyAAQN47rnnWL58OcuXL2fcuHHMnDmT2tpa6urqqK+vZ/369SxbtowlS5Zw2GGHFfhuzKyS+YqbmSXj0LNvKNlr9zn4wxxw6OHE66+z+9vfwynXL+Cp+85np7cMZbeaQ1rV/fNf/sYnv/dbdn5LIwCrdx3OrnsNQX36Un3UxznsvP8uSYwLrjilJK9rZulw4mZmBgwYNpoBw0a3Ktvn3R9ps+4/TvpKq+O9x9Wx97i6NuuamXUn3yo1MzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NEOHEzMzMzS4QTNzMzM7NElDRxk7SbpFsl/UnSY5IOlzRI0hxJS/LvA/O6knSVpKWSFks6pMXrTM7rL5E0uZQxm5mZmfVUpb7i9j1gVkS8DRgNPAacB8yNiOHA3PwY4HhgeP51OnANgKRBwEXAWOAw4KLmZM/MzMysNylZ4iZpV+A9wDSAiNgQEc8DJwDT82rTgQ/lj08AbojM/cBukvYGJgBzImJtRKwD5gDHlSpuMzMzs56qlFfchgGrgZ9IWijpx5J2BvaKiKcB8u975vX3BVa0eP7KvGxr5a1IOl1Sg6SG1atXd/+7MTMzMytYKRO3KuAQ4JqIOBh4mc23RduiNspiG+WtCyKujYjaiKgdPHhwZ+I1MzMz69FKmbitBFZGxLz8+FayRO5v+S1Q8u/Ptqg/pMXzq4GntlFuZmZm1quULHGLiGeAFZJG5EXHAI3ATKB5ZuhkYEb+eCZwSj67dBzwQn4rdTYwXtLAfFLC+LzMzMzMrFepKvHrfw74b0n9gL8CnyJLFm+RdBrwJHBiXvc24P3AUuCVvC4RsVbSpcADeb2vR8TaEsdtZmZm1uOUNHGLiEVAbRunjmmjbgBTtvI61wHXdW90ZmZmZmnxzglmZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZmZmiXDiZmZmZpYIJ25mZtZls2bNYsSIEdTU1DB16tQtzl955ZWMHDmSgw46iGOOOYYnnnhi07lzzz2XAw88kAMPPJCbb765nGGbJceJm5mZdUlTUxNTpkzh9ttvp7GxkZtuuonGxsZWdQ4++GAaGhpYvHgxEydO5JxzzgHgt7/9LQ8++CCLFi1i3rx5XHHFFbz44otFvA2zJDhxMzOzLpk/fz41NTUMGzaMfv36MWnSJGbMmNGqzlFHHcVOO+0EwLhx41i5ciUAjY2NvPe976Wqqoqdd96Z0aNHM2vWrLK/B7NUlDRxk7Rc0sOSFklqyMsGSZojaUn+fWBeLklXSVoqabGkQ1q8zuS8/hJJk0sZs5mZbZ9Vq1YxZMiQTcfV1dWsWrVqq/WnTZvG8ccfD8Do0aO5/fbbeeWVV3juuee48847WbFiRcljNktVVRl+xlER8VyL4/OAuRExVdJ5+fG5wPHA8PxrLHANMFbSIOAioBYIYIGkmRGxrgyxm5lZOyJiizJJbdb96U9/SkNDA3fffTcA48eP54EHHuCd73wngwcP5vDDD6eqqhx/mszSVMSt0hOA6fnj6cCHWpTfEJn7gd0k7Q1MAOZExNo8WZsDHFfuoM3MrG3V1dWtrpKtXLmSffbZZ4t6v/vd77jsssuYOXMm/fv331R+/vnns2jRIubMmUNEMHz48LLEbZaiUiduAdwhaYGk0/OyvSLiaYD8+555+b5Ay+vjK/OyrZW3Iul0SQ2SGlavXt3Nb8PMzLZmzJgxLFmyhGXLlrFhwwbq6+upq6trVWfhwoWcccYZzJw5kz333HNTeVNTE2vWrAFg8eLFLF68mPHjx5c1frOUlPp69Lsi4ilJewJzJP1pG3Xbuq4e2yhvXRBxLXAtQG1t7ZbX7c3MrCSqqqq4+uqrmTBhAk1NTZx66qmMGjWKCy+8kNraWurq6jj77LP5+9//zoknngjAfvvtx8yZM3nttdc44ogjANh111356U9/6lulZtugtsYmlOQHSRcDfwf+FTgyIp7Ob4XeFREjJP1X/vimvP7jwJHNXxFxRl7eql5bamtro6GhoZRvx8wKcOjZNxQdQqEWXHFKl57v9uta+5mViqQFEVHbkbolu1UqaWdJb25+DIwHHgFmAs0zQycDzXPGZwKn5LNLxwEv5LdSZwPjJQ3MZ6COz8vMzMzMepVSXo/eC/hVPrOoCvhZRMyS9ABwi6TTgCeBE/P6twHvB5YCrwCfAoiItZIuBR7I6309ItaWMG4zMzOzHqlkiVtE/BUY3Ub5GuCYNsoDmLKV17oOuK67YzQzMzNLiXdOMDMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0uEEzczMzOzRDhxMzMzM0tEyRM3SX0lLZT0m/x4f0nzJC2RdLOkfnl5//x4aX5+aIvX+Epe/rikCaWO2czMzKwnKscVty8Aj7U4/hbwnYgYDqwDTsvLTwPWRUQN8J28HpJGApOAUcBxwA8l9S1D3GZmZmY9SkkTN0nVwD8BP86PBRwN3JpXmQ58KH98Qn5Mfv6YvP4JQH1ErI+IZcBS4LBSxm1mZmbWE5X6itt3gXOA1/Pj3YHnI2JjfrwS2Dd/vC+wAiA//0Jef1N5G88xMzMz6zVKlrhJ+gDwbEQsaFncRtVo59y2ntPy550uqUFSw+rVq7c7XjMzM7OerpRX3N4F1ElaDtST3SL9LrCbpKq8TjXwVP54JTAEID8/AFjbsryN52wSEddGRG1E1A4ePLj7342ZmZlZwUqWuEXEVyKiOiKGkk0u+H1EfAK4E5iYV5sMzMgfz8yPyc//PiIiL5+UzzrdHxgOzC9V3GZmZmY9VVX7VbrduUC9pG8AC4Fpefk04EZJS8mutE0CiIhHJd0CNAIbgSkR0VT+sM3MzMyKVZbELSLuAu7KH/+VNmaFRsSrwIlbef5lwGWli9DMzMys5/POCWZmZmaJ6FDiJmluR8rMzMzMrHS2eatU0o7ATsAekgayeWmOXYF9ShybmZmZmbXQ3hi3M4AvkiVpC9icuL0I/KCEcZmZmZnZG2wzcYuI7wHfk/S5iPh+mWIyMzMzszZ0aFZpRHxf0juBoS2fExE3lCguMzMzM3uDDiVukm4E3gosAprXUAvAiZuZmZlZmXR0HbdaYGS+k4GZmZmZFaCj67g9AryllIGYmZmZ2bZ19IrbHkCjpPnA+ubCiKgrSVRmZmZmtoWOJm4XlzIIMzMzM2tfR2eV3l3qQMzMzMxs2zq65dVLkl7Mv16V1CTpxVIHZ2YdN2vWLEaMGEFNTQ1Tp07d4vyVV17JyJEjOeiggzjmmGN44oknNp2bPn06w4cPZ/jw4UyfPr2cYZuZ2XboUOIWEW+OiF3zrx2BfwauLm1oZtZRTU1NTJkyhdtvv53GxkZuuukmGhsbW9U5+OCDaWhoYPHixUycOJFzzjkHgLVr13LJJZcwb9485s+fzyWXXMK6deuKeBtmZtaOjs4qbSUifg0c3c2xmFknzZ8/n5qaGoYNG0a/fv2YNGkSM2bMaFXnqKOOYqeddgJg3LhxrFy5EoDZs2dz7LHHMmjQIAYOHMixxx7LrFmzyv4ezMysfR1dgPcjLQ77kK3r5jXdzHqIVatWMWTIkE3H1dXVzJs3b6v1p02bxvHHH7/V565atap0wZqZWad1dFbpB1s83ggsB07o9mjMrFPaWhtbUpt1f/rTn9LQ0MDdd9+93c81M7NidXRW6adKHYiZdV51dTUrVqzYdLxy5Ur22WefLer97ne/47LLLuPuu++mf//+m5571113tXrukUceWeqQzcysEzo6q7Ra0q8kPSvpb5J+Iam61MGZWceMGTOGJUuWsGzZMjZs2EB9fT11da3Xx164cCFnnHEGM2fOZM8999xUPmHCBO644w7WrVvHunXruOOOO5gwYUK534KZmXVAR2+V/gT4GXBifvzJvOzYUgRlZtunqqqKq6++mgkTJtDU1MSpp57KqFGjuPDCC6mtraWuro6zzz6bv//975x4YvYx3m+//Zg5cyaDBg3iggsuYMyYMQBceOGFDBo0qMi3Y2ZmW6GO7BsvaVFEvKO9sp6itrY2Ghoaig7DrE2Hnn1D0SEUZsEVp3Tp+b257cDt11VdbT+zUpG0ICJqO1K3o8uBPCfpk5L65l+fBNZ0PkQzMzMz214dTdxOBU4CngGeBiYCnrBgZmZmVkYdHeN2KTA5ItYBSBoEfJssoTMzMzOzMujoFbeDmpM2gIhYCxxcmpDMzMzMrC0dTdz6SBrYfJBfcevo1TozMzMz6wYdTb7+A/ijpFvJtro6CbisZFGZmZmZ2RY6unPCDZIayDaWF/CRiGgsaWRmZmZm1kqHb3fmiZqTNTMzM7OCdHSMm5mZmZkVrGSJm6QdJc2X9JCkRyVdkpfvL2mepCWSbpbULy/vnx8vzc8PbfFaX8nLH5fkTRTNzMysVyrlFbf1wNERMRp4B3CcpHHAt4DvRMRwYB1wWl7/NGBdRNQA38nrIWkkMAkYBRwH/FBS3xLGbWZmZtYjlSxxi8zf88Md8q8gm+Bwa14+HfhQ/viE/Jj8/DGSlJfXR8T6iFgGLAUOK1XcZmZmZj1VSce45fuaLgKeBeYAfwGej4iNeZWVwL75432BFQD5+ReA3VuWt/Gclj/rdEkNkhpWr15dirdjZmZmVqiSJm4R0RQR7wCqya6SHdBWtfy7tnJua+Vv/FnXRkRtRNQOHjy4syGbmZmZ9VhlmVUaEc8DdwHjgN0kNS9DUg08lT9eCQwByM8PANa2LG/jOWZmZma9RilnlQ6WtFv++E3A+4DHgDuBiXm1ycCM/PHM/Jj8/O8jIvLySfms0/2B4cD8UsVtZmZm1lOVcr/RvYHp+QzQPsAtEfEbSY1AvaRvAAuBaXn9acCNkpaSXWmbBBARj0q6hWzx343AlIhoKmHcZmZmZj1SyRK3iFgMHNxG+V9pY1ZoRLwKnLiV17oM741qZmZmvZx3TjAzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLhBM3MzMzs0Q4cTMzMzNLRMkSN0lDJN0p6TFJj0r6Ql4+SNIcSUvy7wPzckm6StJSSYslHdLitSbn9ZdImlyqmM3MzMx6slJecdsIfDkiDgDGAVMkjQTOA+ZGxHBgbn4McDwwPP86HbgGskQPuAgYCxwGXNSc7JmZmZn1JiVL3CLi6Yh4MH/8EvAYsC9wAjA9rzYd+FD++ATghsjcD+wmaW9gAjAnItZGxDpgDnBcqeI2MzMz66nKMsZN0lDgYGAesFdEPA1ZcgfsmVfbF1jR4mkr87Ktlb/xZ5wuqUFSw+rVq7v7LZiZmZkVruSJm6RdgF8AX4yIF7dVtY2y2EZ564KIayOiNiJqBw8e3LlgzczMzHqwkiZuknYgS9r+OyJ+mRf/Lb8FSv792bx8JTCkxdOrgae2UW5mZmbWq5RyVqmAacBjEXFli1MzgeaZoZOBGS3KT8lnl44DXshvpc4GxksamE9KGJ+XmZmZmfUqVSV87XcBJwMPS1qUl30VmArcIuk04EngxPzcbcD7gaXAK8CnACJiraRLgQfyel+PiLUljNvMzMysRypZ4hYR99H2+DSAY9qoH8CUrbzWdcB13RedmZmZWXq8c4KZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIpy4mZmZmSXCiZuZmZlZIkqWuEm6TtKzkh5pUTZI0hxJS/LvA/NySbpK0lJJiyUd0uI5k/P6SyRNLlW8ZmZmZj1dKa+4XQ8c94ay84C5ETEcmJsfAxwPDM+/TgeugSzRAy4CxgKHARc1J3tmZmZmvU3JEreIuAdY+4biE4Dp+ePpwIdalN8QmfuB3STtDUwA5kTE2ohYB8xhy2TQzMzMrFco9xi3vSLiaYD8+555+b7Aihb1VuZlWys3MzMz63V6yuQEtVEW2yjf8gWk0yU1SGpYvXp1twZnZmZm1hOUO3H7W34LlPz7s3n5SmBIi3rVwFPbKN9CRFwbEbURUTt48OBuD9zMzMysaOVO3GYCzTNDJwMzWpSfks8uHQe8kN9KnQ2MlzQwn5QwPi+zCjVr1ixGjBhBTU0NU6dO3eL8+vXr+ehHP0pNTQ1jx45l+fLlrc4/+eST7LLLLnz7298uU8RmZmblU8rlQG4C/hcYIWmlpNOAqcCxkpYAx+bHALcBfwWWAj8CPgsQEWuBS4EH8q+v52VWgZqampgyZQq33347jY2N3HTTTTQ2NraqM23aNAYOHMjSpUs566yzOPfcc1udP+usszj++OPLGbaZmVnZVJXqhSPiY1s5dUwbdQOYspXXuQ64rhtDsx5q/vz51NTUMGzYMAAmTZrEjBkzGDly5KY6M2bM4OKLLwZg4sSJnHnmmUQEkvj1r3/NsGHD2HnnnYsI38zMrOR6yuQEM1atWsWQIZuHNFZXV7Nq1aqt1qmqqmLAgAGsWbOGl19+mW9961tcdNFFZY3ZzKyrPETEtocTN+sxsguvrUnqUJ2LLrqIs846i1122aVk8ZmZdTcPEbHtVbJbpWbbq7q6mhUrNi/bt3LlSvbZZ58261RXV7Nx40ZeeOEFBg0axLx587j11ls555xzeP755+nTpw877rgjZ555ZrnfhplZh3mIiG0vX3GzHmPMmDEsWbKEZcuWsWHDBurr66mrq2tVp66ujunTs803br31Vo4++mgkce+997J8+XKWL1/OF7/4Rb761a86aTOzHs9DRGx7+YqbbbdDz76hZK/d5+APc8ChhxOvv87ub38Pp1y/gKfuO5+d3jKU3WoO4fWN/Vl+54NcP3Av+u64M/t/4LNbxPPUHx6ib7/+3PS30sS54IpTSvK6Ztb7eIiIbS8nbtajDBg2mgHDRrcq2+fdH9n0uE9VP4bVbftK2j7v+nBJYjMz624eItJ1s2bN4gtf+AJNTU18+tOf5rzzzmt1fv369ZxyyiksWLCA3XffnZtvvpmhQ4cyf/58Tj/9dCBLji+++GI+/OGe//fDiZuZmVlBWg4R2Xfffamvr+dnP/tZqzrNQ0QOP/zwLYaINLv44ovZZZddel3S1jy5Y86cOVRXVzNmzBjq6upajRFsObmjvr6ec889l5tvvpkDDzyQhoYGqqqqePrppxk9ejQf/OAHqarq2amRx7iZmZkVpKqqiquvvpoJEyZwwAEHcNJJJzFq1CguvPBCZs6cCcBpp53GmjVrqKmp4corr2xzyZDequXkjn79+m2a3NHSjBkzmDw527Rp4sSJzJ07l4hgp5122pSkvfrqq1vcou6penZaaWZm1kOUcnzvm0/4GgC/fB5+efYNQA2/vfd5Lrk3/5lDP8iAoR+kCTjxmvuA+97wCsPgZbiphDH2xPG9bU3umDdv3lbrtJzcscceezBv3jxOPfVUnnjiCW688cYef7UNfMXNzMzMEtWVyR0AY8eO5dFHH+WBBx7g8ssv59VXXy1NoN3IiZuZmZklaXsmdwCtJne0dMABB7DzzjvzyCOPlD7oLnLiZmZmZknqyvqfy5YtY+PGjQA88cQTPP744wwdOrTcb2G79fybuWZmZmZtaDm5o6mpiVNPPXXT5I7a2lrq6uo47bTTOPnkk6mpqWHQoEHU19cDcN999zF16lR22GEH+vTpww9/+EP22GOPgt9R+5y4mZmZWcn1xMkdO75/876vl/7hRS79Q89fuN23Ss3MzMwS4cTNzMzMLBFO3MzMzMwS4cTNzMzMLBFO3MzMzMwS4cTNzMzMLBFO3MzMzMwS4cTNzMzMLBFO3MzMzMwS4cTNzMzMLBFO3MzMzMwS4cStm82aNYsRI0ZQU1PD1KlTtzi/fv16PvrRj1JTU8PYsWNZvnz5pnOXX345NTU1jBgxgtmzZ5cxajMzM0uBE7du1NTUxJQpU7j99ttpbGzkpptuorGxsVWdadOmMXDgQJYuXcpZZ53FuedmG9w2NjZSX1/Po48+yqxZs/jsZz9LU1NTEW/DzMzMeignbt1o/vz51NTUMGzYMPr168ekSZOYMWNGqzozZsxg8uTJAEycOJG5c+cSEcyYMYNJkybRv39/9t9/f2pqapg/f34Rb8PMzMx6KCdu3WjVqlUMGTJk03F1dTWrVq3aap2qqioGDBjAmjVrOvRcMzMz692SSdwkHSfpcUlLJZ1XdDxtiYgtyiR1qE5HnmtmZma9WxKJm6S+wA+A44GRwMckjSw2qi1VV1ezYsWKTccrV65kn3322WqdjRs38sILLzBo0KAOPdfMzMx6tyQSN+AwYGlE/DUiNgD1wAkFx7SFMWPGsGTJEpYtW8aGDRuor6+nrq6uVZ26ujqmT58OwK233srRRx+NJOrq6qivr2f9+vUsW7aMJUuWcNhhhxXxNszMzKyHqio6gA7aF1jR4nglMLagWLaqqqqKq6++mgkTJtDU1MSpp57KqFGjuPDCC6mtraWuro7TTjuNk08+mZqaGgYNGkR9fT0Ao0aN4qSTTmLkyJFUVVXxgx/8gL59+xb8jszMzKwnUVtjq3oaSScCEyLi0/nxycBhEfG5FnVOB07PD0cAj5eWfahYAAAgAElEQVQ90I7bA3iu6CAS5vbrGrdf57ntusbt1zVuv87r6W33DxExuCMVU7nithIY0uK4GniqZYWIuBa4tpxBdZakhoioLTqOVLn9usbt13luu65x+3WN26/zKqntUhnj9gAwXNL+kvoBk4CZBcdkZmZmVlZJXHGLiI2SzgRmA32B6yLi0YLDMjMzMyurJBI3gIi4Dbit6Di6SRK3dHswt1/XuP06z23XNW6/rnH7dV7FtF0SkxPMzMzMLJ0xbmZmZma9nhM3MzMzs0Q4cSsTSW+SNKLoOFLl9rOiSKqWdFT+uL+knYuOKSWS+kmqKTqOlEnqX3QMKarUvufErQwkfRBYBMzKj98hycuZdJDbr2uU+aSkC/Pj/SR5P7UOkHQq2dJDP86L/gGYUVxEaZH0T8DDwJz8+B2SflVsVOmQdJikh4El+fFoSd8vOKwkVHLfc+JWHheT7bf6PEBELAKGFhhPai7G7dcVPwQOBz6WH78E/KC4cJLyeWAc8CJARPwZ2LPQiNLydbLtCVt+divuCkgJXQV8AFgDEBEPAUcVGlE6KrbvOXErj40R8ULRQSTM7dc1YyNiCvAqQESsA/oVG1IyXo2IDc0HkvoCKjCe1LwWEc+/ocxLGXRcn4h44g1lTYVEkp6K7XtO3MrjEUkfB/pKGp5f6v5j0UElxO3XNa/lCUcASBoMvF5sSMn4g6RzgB3zcW43A78pOKaUPCbpJKBPvvPNd4H7iw4qISvyYQ0hqa+kLwJ/LjqoRFRs33PiVh6fA0YB64GfAS8AXyg0orS4/brmKuBXwJ6SLgPuAy4vNqRknEN2a/lPZH1uLvDVQiNKy5nAoWT/KPyS7KqvP7sd92/Al4D9gL+R3bb/TKERpaNi+54X4C0DSSdGxM/bK7O2uf26TtLbgGPIbvPNjYjHCg4pCZLOjIir2yuztkn6SET8sr0ya5ukcRFxf3tltqVK7ntO3MpA0oMRcUh7ZdY2t1/XSLoxIk5ur8y2tJW+tzAiDi4qppRspf0WRMShRcWUErdf51Vy2yWzV2mKJB0PvB/YV9JVLU7tCmwsJqp0uP26zaiWB/l4t+R/eZWSpI8Ck4D9JbX8D/3N5LPUbOskTQCOI/vsXtni1K54fGW78nFthwODJX2+xaldgR2KiSoNvaHvOXErraeABqAOWNCi/CXgrEIiSovbrwskfYVsPNabJL3I5tmQG6igDZdLZD7ZEgzVtF465SVgYSERpeVZ4BGycUWPtih/CTivkIjSsjOwB9nf6MEtyl8CTiwkonRUfN/zrdIykLRDRLxWdBypcvt1jaTLI+IrRcdhvY+kHSPi1aLjSJWkYRHx16LjSFEl9z0nbmUgaTjZLL6RwI7N5RExrLCgEuL26zpJA4HhtG6/e4qLKA2SxgDfBw4A+pNdtVwfEbsWGlgiJL0VuIwtP7v/WFhQCZG0B/BlsuEOLdtvfGFBJaKS+56XAymPnwDXkI3LOgq4Abix0IjS4vbrAkmfBu4BZgOX5N8vLjKmhPwQmAz8lWx825nAdwuNKC3Xk31+BRwP3ALUFxlQYn4KLAf+EfgW8AzZ9n/Wvuup0L7nxK083hQRc8mucD4RERcDRxccU0rcfl3zBWAM8EREHAUcDKwuNqRk9ImIx4GqiHgtIn4EvK/ooBKyU0TMBoiIv0TE1/CWTdtjcET8F7Ah/x04mWz7P2tfxfY9T04oj1cl9QGWSDoTWIX3O9webr+ueTUiXpWEpP4R8SdJI4oOKhEvS+oHPCTpm8DTwC4Fx5SS9ZIE/EXSZ/Bnd3s1j+19Jp8t+RQwpMB4UlKxfc9j3MogHyfzGLAbcCkwAPh3L6LYMW6/rpH0K+BTwBfJrlSuA3aIiPcXGlgCJA0j+2O5I9lYowHA1flm89YOSWOBRmAg2XijAcC3IuIPhQaWCEl1wN3AP5DNbt4VuKQSFpEttUrue07czHoRSe8l+wU2q+Xm6WZmlgYnbiUk6X/IN/ZuS0TUlTGc5Lj9ukbSoG2dj4i15YolNZIWsu2+5107tiG/yrut9vtIGcNJjqTvsO32+1IZw0lKb+h7HuNWWt/Ov38EeAvZDCGAj5HNFLJtc/t1zQKyX2Ai26R6Xf54N+BJYP/iQuvxJubfPwP0ZfMs5k+QLeRp29a8l+sJwD7Af+fHHwP+UkhEaXkk/z4OOJBsRiRk/fKBQiJKR8X3PV9xKwNJ90TEe9ors7a5/bpG0n8CMyPitvz4eOB9EfHlYiPr+ST9ISLe1V6Zte2Nn9N8sPjd/ux2jKTfAxOaFyDPJ8rMigjPqm9HJfc9LwdSHoPzQc4ASNqf1tuY2La5/bpmTHPSBhARtwPvLTCelOwiaVzzQT7g2bNKO25PSUNbHO+HP7vbY1+y7a+a7ZSXWfsqtu/5Vml5nAXcJal565KhwOnFhZMct1/XPCfpa2S3mgP4JNk+nNa+TwM/kbQjWdu9CpxabEhJ+TJwr6TH8+PhZLefrWOuABZJ+l1+fDTwjQLjSUnF9j3fKi0TSf2Bt+WHf4qI9UXGkxq3X+flkxQuAt5DlnzcA3zdkxM6TtLuABHhhHc7SXoT2bZDAI0R8X9FxpMaSfuSjXUDuD8iVhUZT0oqte85cTMzMzNLhMe4mZmZmSXCiZuZmZlZIjw5oUwkHUQ2qH5Tm3vbko5z+1kR8j1yj2PLvndVUTGlRtJItmy/mYUFlBhJuwLVtG6/xcVFlI5K7XtO3MpA0nXAQcCjwOt5cQBOPDrA7dc1+fIpn2PLX2DeeaJ9M8j62sNs7nvWQZJ+BNSS7RnZ8rOb/B/PcpB0EdkM+mVs3g0gyCYa2TZUct/z5IQykNQYESPbr2ltcft1jaSHgGm8IfmIiLsLCyoRkh6OiLcXHUeqJD0GjAz/oemUfCmLgzyLfvtVct/zFbfy+F9JIyOisehAEuX265pXfWuv02ZLOjoifl90IImaB/wj8Hh7Fa1NjwJvBpy4bb+K7Xu+4lYGkt4D/A/wDNkHUEBExEGFBpYIt1/XSPo42eKTd9DiD0BEPFhYUImQdALwM7JbLBvY3PcGFRpYIiQdQfbZXUXrz+4hhQaWCEmHAr8GFtP6s5v8RumlVsl9z1fcyuM64GQ8Tqaz3H5d83ay9jua1mM9vN9h+74DHIH7XmddR7bThNuvc6aT9UG33/ar2L7nxK08nqyEmSwFcvt1zYeBYRGxoehAErQEWFiJ42TKZIVnf3fJ2oi4suggElWxfc+JW3n8SdLPyC7btrzcXZGdqgTcfl3zELAb8GzRgSToKeD3km6jdd/zmMGOaZR0A1t+dv2PWMc8IOlSspmQLdvPy4G0r2L7nhO38ngTWccZ36LMy1l0nNuva/YiS34foPUvMC8H0r6V+deuRQeSqAH595Z9rSKWZCiTw/LvR7Yo83IgHVOxfc+JW4lJ6gssjojvFB1Litx+3eKiogNIUd73doiI84qOJUV5+z3gq5Odk7ffdyPiF0XHkppK73ueVVoGku6MiKOKjiNVbr/Oy3+BzY6I9xUdS4okzY2IY4qOI1WS7oqII4uOI1WS7o2II4qOI0WV3PecuJWBpMvILtveDLzcXO7lGDrG7dc1kmYCJ0fEC0XHkhpJ3waGAT+ndd9L/nZLOUj6Btk6ZPW0bj+P0eoASV8D/s6Wv/teLCyoRFRy33PiVgaS7myjOCLCyzF0gNuvayTdAowD5tD6F9jnCwsqEZJubKM4IuKUsgeTIEn3tlEcEeExWh0gaUWLw2DzWmT7FRRSMiq57zlxM6twkia3VR4R08sdi5mZdU2fogPoDSTtJWmapNvz45GSTis6rlS4/bomT9BuAe6PiOnNX0XHlQJJNZJm5/u9IukgSV8pOq5USBos6b8k/SY/HinpXwoOKxmS3iTpPEnX5Mc1ko4vOq4UVHLfc+JWHtcDs4F98uM/A18sLJr0XI/br9MkfRBYBMzKj9+Rj3uz9v0YuITNK68/DHyyuHCScz1wNzAkP14CfLmwaNJzHdnf6eYJCk8B3ywunKRcT4X2PSdu5bFHRNxC/ss/IjYCTcWGlBS3X9dcTLYe1PMAEbEI2L/IgBKyc0T8sfkg30HhtQLjSc2eEfEzNn92X8Of3e0xPCK+Sd7nIuIVsnFu1r6K7XtO3MrjZUm7kw0uRdI4wDP8Os7t1zUb25hR6sGtHbNG0v5s7nsfAp4pNqSkvCxpEJvbbwzwUrEhJWWDpB3Z3H77A966rmMqtu95Ad7y+BLZas1vlfQHYDBwYrEhJcXt1zWPSPo40FfScODzwB/beY5lzgSmAW+T9ATwNDCp2JCScjbZlkPDJN0N7Is/u9vjUrIhDtWSpgPvBT5dbEjJqNi+51mlZSCpP9kl2hFkl7kfB/pExPptPtEAt19XSdoJOJ/NW4bNBi51+7VP0n4R8aSkAWS/L59vLis6thRIqiK7s3MA2We3EXg9H+5gHSBpMPBOsvb7Y0R4z+EOqOS+58StDCQ9GBGHtFdmbXP7dY2kEyPi5+2V2Zbc97rG7dc1ku6IiPHtldmWKrnv+VZpCUl6C9nl2TdJOpjNg0p3BXYqLLBEuP26zVfIVv5vr8xykv6R7D/1AZJablK9K7BjMVGlQ9KewN5kn92348/udpHUj6yf7SXpzbRuPy++uw29oe85cSutCcC/ANXAf7C5A70IfLWgmFLi9uuCfL2n9wP7Smq52fKuQPK3C0psFPARYDdaj4t5CTijkIjS8k/AqWSf3R/Q+rN7QVFBJWQK2djePYFHad1+/1lUUImo+L7nW6VlIOmfI+IXRceRKrdf50gaDbwD+DpwYYtTLwF3RsS6QgJLiKR3R8R9RceRKkkn5Uv5WCdI+mJEfLfoOFJUyX3PiZtZhZO0Q76GkZmZJc6Jm5mZmVkivABviUnqI+mdRceRKrefmZnZZk7cSiwiXicbWG+d4PbrPpJ2LjqG1FTyRtXlkG+S/hVJ/5kfe5P07SRpkqTz88dDJB1adEwpqOS+58StPO6Q9M+SvMdc57j9ukDSOyU1Ao/lx6Ml/bDgsFJxPRW6UXWZXEc2q+/d+bE3Sd8Okq4GjgI+mRe9jGeVdlTF9j0nbuXxJbI1szZIelHSS5JeLDqohLj9uuY7ZEurrAGIiIeA9xQaUToqdqPqMvEm6V3zzog4A3gVICLWAv2KDSkZFdv3vI5bGUTEm4uOIWVuv66LiBVvuGDp5KNjKnaj6jLxJuld85qkPmxuv93J/4mwdlVs33PiVgb5Lb5PAPtHxKWShgB7R8T8gkNLgtuvy1bkEzwiX5H98+S3Ta1d/48tN6qeWGxISfk6W26SflqxISXlB8AvgMGSLgFOAi4pNqRkVGzf83IgZSDpGrL/ko6OiAMkDQTuiIgxBYeWBLdf10jaA/ge8D6yWwV3AF+IiDWFBtbD5Vc6xgALabFRdURUxH/tpZb/w/UWsl06vEl6J0kaxebP7u8i4pGCQ+rxKr3vOXErg+aNbSUtjIiD87KHImJ00bGlwO3XeZL6Ap+PiO8UHUuKJN0fEeOKjiNVkhZEhGdBdkL+2X3Qv+c6p5L7nicnlMdr+Yew+V77YDxOYXu4/TopIpqAE4qOI2FzJLn9Om++pEOKDiJF+We3UdK+RceSqIrte77iVgaSPgF8FDgEmE42RuaCSt1Hrbu5/bpG0mXAAOBmsuUEAIiIBwsLKhGS1pG13Xrg/8huuUREDCo0sERIepjsNvNfyPpec/tV5B/U7iZpDjAW+F9af3Y/UlhQiajkvufErUwkvQ04hqzzzI0IDw7fDm6/zpN0ZxvFERFHlz2YxORXereQXw2xdkh6a1vlEfGXcseSIknHtFUeEXPLHUtqKrnvOXErA0k3RsTJ7ZVZ29x+ViRJ72fzund3RcSsIuNJjaQD2bwI6r0R8WiR8aQmn1xUmx82RMRzRcaTkkrtex7jVh6jWh7k/8VX5KDJEnH7dYGkAZKulNSQf/2HpAFFx5WC/DbzOcBf869zJH2j2KjSIelM4BZgv/zrFkmfLTaqdEj6Z+BB4GTgFKBB0oeLjSoNldz3fMWthCR9Bfgq8CbgleZiskUAr42IrxQVWwrcft1D0i+AR8jGB0L2R2C0x8m0T9Ji4ODmW6OSqshm+h1UbGRpyNvvnRHx9/x4F7JlGdx+HSDpIWB8RPwtP96LbCkkzzRtRyX3PS/AW1r3RMTlkqZGxHlFB5Mgt1/3eGtE/HOL40skLSosmvTsCqzLH3sXj+0j8i2Hcq9RIdsOlUmf5qQttxrfKeuoiu17TtxK6yqyW3rjASce28/t1z3+T9K7I+I+AEnvIpshae37d+BBSXPJfukfCVxYaERpuRG4P7/qC/BhNl/5tfbNkXQb8LP8eBIwu8B4UlKxfc+3SktI0v1kWwu9n2wphlYi4vNlDyohbr/uIWk0cAPZshaQXT2aHBGLi4uqZ5M0LiLuz2+N7kW2JIOA+yNiVbHR9XyS9ouIJ/PHY4AjyNrvnoh4oNDgEiCpKiI25jsAnEg2wF7APcCt4T/cW9Ub+p6vuJXWB8i2KjkaWFBwLCly+3WBpC9ExPeAXSJitKRdASLixYJDS8EPyK72zs/XffplwfGk5lfAoZLuiIjxQEX8wSyjeWT97ycR8S9kg+ytYyq+7/mKWxlIGh0RDxUdR6rcfp0jaVFEvKN5y7Ci40mJpHnAYqAO+O83no+IL5U9qITkYyh/DnwGuOKN5yPiqrIHlRBJjwCXk22UftYbz0fEzLIHlYje0Pd8xa08Hpc0hWxZix2bCyPi1OJCSorbr3Mek7QcGJzPsGrWvIJ48rOrSuiDZGMrxwMVsfZTmX0M+AjZ35jBBceSoinAJ4HdyG6VthSAE7etq/i+5ytuZSDp58CfgI+T/Qf1CeCxiPhCoYElwu3XeZLeQjaYue6N5yLiifJHlBZJh0aEb9N3kqQPRsT/FB1HqiSdERH/VXQcKarkvudpxeVRExEXAC9HxHTgn4C3FxxTStx+nRQRz+RrPj0L7BgRTzR/FR1bIl6QNDtfTwtJB+XrC1rHzJP0X5J+AyBppKR/KTimlNwo6TxJ1wBIqpF0fNFBJaJi+54Tt/JoXkvm+XwLjgHA0OLCSY7brwskfRBYBMzKj98hybdaOubHwCXA6/nxw2S3sKxjfgLcDQzJj5cAXy4unORMI/s7fUR+/BTwzeLCSUrF9j0nbuVxraSBwNfIxiY0At8qNqSkuP265mLgMOB5gIhYhBPfjto5Iv7YfJAvw/DaNupba3tGxM/IE9+IeA1oKjakpAyPiG+S97mIeIUKWUS2DCq273lyQhlExI/zh/cAw4qMJUVuvy7bGBEvZEtC2XZaI2l/sgHhSPoQ8EyxISXlZUmD2Nx+Y4CXig0pKRsk7cjm9tufbMs/a1/F9j0nbmaV7xFJHwf6ShoOfB74YzvPscyZZLer3ibpCeBpstXrrWP+H/A/wDBJdwP7AhOLDSkpXycb4lAtaTrwXuC0YkNKRsX2Pc8qNatwknYCzidb2kJks0wvjYhXCw0sIZIGkP2+fL7oWFIjqR9wAFnfa4wIXzHaDpIGA+8ka78/RsSzBYeUjErte07cSkjSrl6lvvPcft0r3zkhIqIibheUQz628gKyLYcCuA/4RkSs2+YTDQBJ/YEz2Nx+9wI/ioj1hQaWEEl1tOh/lbrERXer5L7nyQmltVCSb6t0ntuvG0gaI+lhsp0AHpb0kKRDi44rEfVk42I+QTab9EXa2DfXtmo62dZNPyKboXsIFbLRdzlI+j7wBbIZkUuBz+dl1r6K7Xu+4lZCkv4B+C6wC/BvEbG04JCS4vbrHvmuCVMi4t78+N3AD71zQvskLYiIQ9srs7ZJWvzGfibpoXxtQWuHpEeBA5s3lZfUF1gcEaOKjaznq+S+58kJJZQvcvphSccBf5D0AJvXgyIitljN3jZz+3Wbl5qTNoCIuE+Sb5d2zN2SJkbErQCSPgLcXnBMKVkkaUxEPADZThTA/xYcU0r+DFQDK/LjvYFHigsnKRXb93zFrcQkjQCuAdYCP6B14nF3UXGlwu3XeZKaN5Y/GdgJuIlsrMdHgXURcX5RsfV0ktaRtZXIFnx+LT/uBzwfEYMKDK/Hk7SQze01EvhrfjwMeKQSrnqUkqRfkbXXbmRrMN6fHx8O/CEiJhQYXo/WG/qeE7cSkjSVbI/IL0eE/0vfTm6/rpF05zZOR0QcXbZgEpPfktqqiKiIhTxLRdJbt3U+Iv5SrlhSJOmYbZ2PiLnliiU1vaHv+VZpaTUBh3jZhU5z+3VBRBxVdAypcmLWNZXwx7FITsw6rzf0PV9xK4ikYyNiTtFxpMrtZ2ZmvZETt4JIejIi9is6jlS5/czMrDfyrdISkjRza6eA3csZS4rcfl3jBYw7T9L/kC2h8mTRsaRI0lXAVyPi70XHkiJJXwK+GxGvt1vZWukNfc+JW2kdQbZo5xs7kMhmCtm2uf26ZqGk8yOivuhAEnQTMFfSj4Fve8zbdnsGeFDS1yLilqKDSdAIoEHSZyPi/qKDSUzF9z3fKi0hSbcD/x4RW8zuk3RPRLyngLCS4fbrGi9g3DWS3gxcDBxNtuJ6y6VorioorGRI2g/4Dln/u4bW7be1q+mWkzQG+D7wEFu23+Ki4kpBpfc9J25mFS5fwHg64AWMt4OkKuAcYDJwK63b7oKi4kqJpI8DVwB3sbn9IiJOKSyohEh6D/BroJFsLTLI2s//tLajkvueb5WaVbB8AeNzyDZYbrWAsW1dvo7W94BZZEvSvFxwSEmR9DayKx1rgLERsbLgkJIiaQ+ypOMA4H0R8WDBISWjN/Q9X3ErE0k7kF3x+NfmLTisfZIGAr9suSZZvjDvXRExq7jIej4vYNx5kv4IfMa3pDpH0uPAWRFxW9GxpEjSX4FvA9eE/0hvl7zvfbGSf+f5ilv5nEC2Bce/kiVw1gERsU7Si5KOiIh7JfUHTgQuLDq2BHgB406KiHcWHUPi3hER/9eywLOct8vhEfG3lgWSRkfEQ0UFlJAt+l6l6VN0AL3IacCpwJGSdio6mMT8mKztAD4M3B4RGwqMJwkRcf7WkjZJx5Y7npRIGiXpPknLJP1Q0oAW5ypio+oSGy3pYUkPSRojaTbwiKQnJI0tOrgE7CXpoBZfo4HfSnq7pIOKDq6HO6n5gaR9JM2WtEbSPZKGFxlYd/Gt0jKQNAT4dUQcKunfgcci4idFx5WKfN/IRuAQ4BfAuf7Ps2u8gPG2SboX+BbZ5t6fBj4B1EXEMkkLI+LgQgPs4STNAz5DNqvvl8DEiLhbUi3Z+mTvLjTAHk7S62R3Zlr+g1oLNODJCdsk6cGIOCR/XA/cA1xL9k//GRHxviLj6w6+VVoenwJuyB//BPhR/t06ICKaJP0C+H/AICdtHeMFjLvkzRHxm/zxVEkNwB35TDX/t9u+fhGxEEDSmoi4GyAiGnzHoUM+DvwbcFlE3AEgaVlEHFFsWMl5W0RMyh//XNJXC42mmzhxKzFJIltEdhxARDwmqa+kERHxeLHRJWUa8Cfg80UHkhAvYNx5fVqOyYqI30k6Efg5MLDY0JLQchjO+W8416+cgaQoIuol/Rb4pqRTyf5p9T8MHVMt6Uqy33N7SNohIl7Lz1VEzlMRb6KHezPZDJe1Lco+W1QwqYqIv0j6KHBH0bEk5H7glearHS3lM69s664ARgGbxrNFxKJ8bOBF/7+9+w/2rK7rOP58pQsLLKUmQVSE4viDUBHIneXHAEVOjcIQoGQ7GgxQmIDWMBWZxjAtMaKkMYIDSFtElLVSDDFByk7MVqy7sOmy7CChOCVQmpWQuK3uqz/O57Zf7957997z+d7v2XP29Zj5zu45597vvvc1d+9+7jmfz/vTWVX9caWk/W1/0/aaqZOSjgBu77Cu3rD9LHBpebx8O83/JbF7V4z8/hGa3L4u6RBgECtNM8ctImIeJL3U9te6rqOvkl975cnNi2z/Z9e1RPeyqjQiYn5yt7dO8mup9HK7t+s6+krSZ7uuYZwycIsYMEkvlrR22rlryjZYsTDquoCeS351lnRdQI8NKrsM3CIGrDxa+YakkwBGGhjf32lh/XRr1wX0XPKrk51i2htUdpnjtogkXc8cK4FsZ4XkHJLfeEg6HTjL9vmSfg440fYlXde1J5N0D/DLtp/supY+Sn51JN0A/EZ2mli4vSG73HFbXBuBh4ClNM1jHy+vo2m2I4q5Jb/xuAc4XtIBwHk0fQRjbqtp+ra9r+wzHAuzmuRX40ngodI3MBbmSQaeXe64TUCZY/SmqV4y5RvZfaMbp8fskl89SVcD24A3204Pt3koA90PAD8N3AbsmLpm+7qu6uqL5FdH0g8B1wEvBW7ku/P7VFd19cHQs0sft8k4lNJLphwvK+difpJfvTQwXrjtwP8A+9J8/e2Y+8NjmuRXwfZXShPeVcDp7MzPNNuIxSyGnl0GbpNxDbBpZHXfycCV3ZXTO8mvUhoYL0xZdXsdcBdwjO1vdlxSryS/OpJ+jOZO0VPAG20/3XFJvbE3ZJdHpRNSujYvL4frbT/TZT19k/xiksom8xfb3tJ1LX2U/OpI2kqz4056ty3Q3pBdBm4TULperwRebvsqSYcBh9geVFPAxZL8Yk8iaZnt6fu/RoyNpH1tb5t27iXTtk6MGcyS3Rm27+qqpnHLqtLJuAFYAby9HD8LfKy7cnon+cWe5NGuC9jTSXqtpAcl/YukmyS9eORafuDavWMlbZW0RdJySX8LbCx5rui6uD3cmyWdNfI6G7hp6rjr4sYhc9wmY7ntYyRtgqYpqqR9ui6qR5JfTJSkX53tEs3imJjbjTTzUB8ELgTWlbseTzCwLvaL5CPA22i+1v4aONP2OknHANcDJ3RZ3B7ukzQNd/+dnbt1HECzSCGLE2Letkt6AaWZrKSDyAqrhUh+LaSBcZWrgWuBb89wLU8qdm+Z7alu9R+S9D6vM50AAAkySURBVBDwN5LewRxfk/H/ltjeDCDpq7bXAdh+WNJ+3Za2x1tBs6BtA/Bx25Z0iu3zO65rbDJwm4zfB+4EfkDSKuAc4Le6LalXkl87G8uvJwBHAn9Wjt9K09g4Zvcw8Je2d8lJ0oUd1NM3kvR9tv8bwPba8shqDfCSbkvrhdEfDq6Ydi1PG+Zge4OknwIuBe6X9OsM7IeFLE6YEEmvBn6S5tbtZ2xv7bikXkl+7aWB8cJJehXwH7a/NsO1g23/Wwdl9UbpWv9F2w9OO38Y8H7bF3VTWT9IOgP49PQ2KpKOAM62/cFuKusXSYfSPHY+zvbLu65nXDJwm4Dyj+1fbW+TdArwOuCPbP9Xt5X1Q/KrI+kxYMXUirQyUfxB26/qtrKIiFiozNWYjDXAdyS9ArgFeBnwJ92W1CvJr85UA+PVklbTPAa8utuSIiKijdxxmwBJD5dVkb8GPG/7ekmbbL+h69r6IPnVSwPjiIhhyB23ydgu6e3AO4G7y7ksiZ+/5FehNDA+DXi97b8C9pGUjeYjInooq0on43zgYmCV7S9Jehnwxx3X1CfJr84NNO1TfgK4iqaB8Rrgx7ssak+WVip1kl+d5Nfe3pBd7rhNgO1HgcuBzZKOoplof03HZfVG8qu23Pa7gW9B08CYtBTYnY00LVOWAscAj5fX0cB3OqyrL5JfneTX3uCzyxy3CSgrIf8QeJKmncWPAL9g+4EOy+qN5FdH0nrgeGBDmSt4EE07kMwR3I20UqmT/Ookv/aGnF0elU7Gh2m+gB4DkPRK4A7g2E6r6o/kVycNjNs7FDgQmNrce1k5F/OT/Ookv/YGm10GbpOxZGrQAWD7C2X0H/OT/CrYvr1sOTTVwPjMNDCet6lWKmvL8ck0e3DG/CS/OsmvvcFml0elEyDpVprJkreVUyuBFw5p77TFlPzqpIFxnbRSqZP86iS/9oaaXRYnTMa7gC3AZcB7gEdpVknG/CS/Omlg3FJaqdRJfnWSX3tDzi533CIGLg2M25N0I6WViu3XlO3C7rOdVirzkPzqJL/2hpxd5rgtIkmbmbufzOsmWE7vJL+xGW1gfHo5lzmC87O8DHo3QdNKRVJaqcxf8quT/NobbHYZuC2ut3RdQM8lv/FIA+P2tkt6AeUHiNJKZUe3JfVK8quT/NobbHaZ47a4lgA/bPvLoy/gMDJono/kNwZpYFxleiuVdcDV3ZbUK8mvTvJrb7DZZY7bIpJ0N/Cbtj8/7fxxwG/bPn3mzwxIfuOSBsZ1JL2ana1UPpNWKguT/Ookv/aGml0GbotI0iO2j5rl2mbbr510TX2S/Maj9HD7+ekNjG2ngfFupJVKneRXJ/m1N+Ts8qh0cS2d49p+E6uiv5LfeOzSwJgsTpivtFKpk/zqJL/2BptdBm6La4Oki6aflHQBzSa4MbfkNx4bJX1C0inldTPJb7522P42cBbwUdu/AvxgxzX1SfKrk/zaG2x2meC9uN4L3ClpJTv/ozwO2Af42c6q6o/kNx7vAt5N08BYwAPADZ1W1B9ppVIn+dVJfu0NNrvMcZsASacCU3O1tti+v8t6+ib5RVckHUnTSuUfbd9RWqmcm1W585P86iS/9oacXQZuEQOVBsbjUZp2vrIcPmZ7e5f19E3yq5P82htqdhm4RQyUpB+d63rpiRdzSCuVOsmvTvJrb8jZZeAWMVBlNdXBtv9+2vmTgKdsP9FNZf2RVip1kl+d5NfekLPLqtKI4foI8OwM558v12L30kqlTvKrk/zaG2x2WVUaMVyHT991AsD2RkmHT76cXtoo6RPAbeV4dIVz7F7yq5P82htsdnlUGjFQkv7Z9isWei12krQvTSuVExlppWJ7W6eF9UTyq5P82htydhm4RQyUpDuA+23fPO38BcCbbJ/bTWUREdFWBm4RAyXpYOBO4H+ZoYGx7We6qm1Pl1YqdZJfneTX3t6QXQZuEQOXBsYLl1YqdZJfneTX3t6QXRYnRAyc7bXA2q7r6JklzNFKpZuSeiX51Ul+7Q0+u7QDiYjYVVqp1El+dZJfe4PPLgO3iIhdzdpKBTh88uX0TvKrk/zaG3x2GbhFROxq6RzX9ptYFf2V/Ookv/YGn10GbhERu9og6aLpJ0srlUE08Vxkya9O8mtv8NllVWlExDRppVIn+dVJfu3tDdll4BYRMYu0UqmT/Ookv/aGnF0GbhERERE9kTluERERET2RgVtERERET2TgFhEREdETGbhFRG9I+n5J/1Rez0j6ysjxP4zpzzhP0lclbZL0uKR7JR0/cv0qSaeV358kaUv58/eTdG05vnaO93+npEfKxz0q6fLd1HOmpCPH8XeLiP7L4oSI6CVJVwLP2f7QmN/3POA425eU41OBO4BTbW+d9rEfB9bb/oNy/A3gINvbZnnvnwFWAW+x/ZSkpcA7bN88Rz2rgbtt/0X1Xy4iei933CJiECQ9V349RdLfSfqkpC9IukbSSkmflbRZ0hHl4w6StEbShvI6Yab3tb0WuAn4xfJ5qyWdI+lC4G3AByTdLuku4ABgvaRzZynzCuBy20+V9/7W1KBN0kWljs+VuvYvd/rOAK4td/WOGFdeEdFPL+y6gIiIRfB64DXA14EvArfYfqOk9wCXAu8FPgr8nu11kg4D7i2fM5OHgV8aPWH7FkknMnI3TNJzto+eo66jmL17+6dGBnG/A1xg+/oyIMwdt4gAMnCLiGHaYPtpAElPAPeV85uBU8vvTwOOlDT1Od8r6cBZ3k+znB+no8qA7UXAMpqBZETEd8nALSKGaHSO2Y6R4x3s/L73PcAK28+PfuLIQG7UG4CtM11YoC3AscBMXdxXA2fa/lyZZ3fKGP68iBiYzHGLiL3VfcAlUweSZnzEKelkmvltsy4gWIDfBT4o6ZDy3vtKuqxcOxB4WtISYOXI5zxbrkVE5I5bROy1LgM+JunzNN8LHwAuLtfOLfPX9ge+BJw9fUVpG7bvKZtgf1rNrT0Dt5bL7wfWA1+meaQ7NVj7U+DmMsA7x/YTtXVERH+lHUhERERET+RRaURERERP5FFpRMSYSXof8NZpp//c9qou6omI4cij0oiIiIieyKPSiIiIiJ7IwC0iIiKiJzJwi4iIiOiJDNwiIiIieiIDt4iIiIie+D8OQWA3muZBoQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "time_off = sb.countplot(data = default_off_cat, x = 'TimeDiff_Cat', color = base, order = order_time)\n",
    "plt.xticks(rotation=90)\n",
    "\n",
    "for p in time_off.patches:\n",
    "    height = p.get_height()\n",
    "    time_off.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_default_off),\n",
    "            ha=\"center\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> As most of the loans are 3Y let's focus on those."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_loans_3 = df_loans[df_loans['Term']==36]\n",
    "total_3 = df_loans_3[df_loans_3['TimeDiff_Cat'].notnull()]['ListingNumber'].count()\n",
    "order_time_3 = [\"Closed >1Y after term date\", 'Closed <1Y after term date', 'Closed <1Y before term date', 'Closed 1Y-2Y before term date', \"Closed 2Y-3Y before term date\"]\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "time_3 = sb.countplot(data = df_loans_3, x = 'TimeDiff_Cat', color = base, order = order_time_3)\n",
    "plt.xticks(rotation=90)\n",
    "\n",
    "for p in time_3.patches:\n",
    "    height = p.get_height()\n",
    "    time_3.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_3),\n",
    "            ha=\"center\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "default_off_3 = default_off_cat[default_off_cat['Term']==36]\n",
    "total_default_off_3 = default_off_3[default_off_3['TimeDiff_Cat'].notnull()]['ListingNumber'].count()\n",
    "order_time_3 = [\"Closed >1Y after term date\", 'Closed <1Y after term date', 'Closed <1Y before term date', 'Closed 1Y-2Y before term date', \"Closed 2Y-3Y before term date\"]\n",
    "\n",
    "fig = plt.figure(figsize = (10,5))\n",
    "time_off_3 = sb.countplot(data = default_off_3, x = 'TimeDiff_Cat', color = base, order = order_time_3)\n",
    "plt.xticks(rotation=90)\n",
    "\n",
    "for p in time_off_3.patches:\n",
    "    height = p.get_height()\n",
    "    time_off_3.text(p.get_x()+p.get_width()/2.,\n",
    "            height + 3,\n",
    "            '{:0.2f}'.format(height/total_default_off_3),\n",
    "            ha=\"center\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Various insights can be drawn from the above:\n",
    "- 18% of the overall population of loans is closed after the term final date whereas this is true for 4% of the defaulted/ charged off loans. This is a sign that loans that are meant to default get identified generally earlier than the final date, which is a good thing for Prosper\n",
    "- If we look at 3Y loans only, these values become 19% for the overall population and 5% for the defaulted/charged off ones\n",
    "- 31% of the 3Y loans meant to default or be charged off are closed in the first year. This shows that for almost a third of the loans that ended up being charged off or defaulted, signs of financial distress could be seen quite soon, maybe even at the first due payments. 43% were closed between 1 and 2 years before term date and both percentages are higher than the ones for the overall population. \n",
    "\n",
    "### Credit Rating vs Homeowner vs Loan Status\n",
    "> Here we investigate the relationship between credit rating, being a homeowner and charged off/defaulted loans."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(17000, 15)"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "default_off.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Current                   55157\n",
       "Completed                 37921\n",
       "Chargedoff                11986\n",
       "Defaulted                  5014\n",
       "Past Due (1-15 days)        805\n",
       "Past Due (31-60 days)       363\n",
       "Past Due (61-90 days)       312\n",
       "Past Due (91-120 days)      304\n",
       "Past Due (16-30 days)       265\n",
       "FinalPaymentInProgress      202\n",
       "Past Due (>120 days)         16\n",
       "Cancelled                     5\n",
       "Name: LoanStatus, dtype: int64"
      ]
     },
     "execution_count": 107,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_loans.LoanStatus.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_loans['Rating'] = df_loans.Rating.astype(str)\n",
    "multi = pd.DataFrame(default_off.groupby(['IsBorrowerHomeowner', 'Rating'])['ListingNumber'].count()).reset_index()\n",
    "multi['Rating'] = multi['Rating'].astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [],
   "source": [
    "multi_aggr = pd.DataFrame(df_loans.groupby('Rating')['ListingNumber'].count()).reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "multi_visual = multi.merge(multi_aggr, on = 'Rating', how = 'left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [],
   "source": [
    "multi_visual['Proportion'] = multi_visual['ListingNumber_x']/multi_visual['ListingNumber_y']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmQAAAFACAYAAAASxGABAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3Xuc1XW97/HXhxHFI6amiG5RQMXSQJFrHg8IaUqdDopuE7sox1tmbCuPbOvsDsfNrp1mu6gdWzMvZamgtDV2UeY1tUwBxVIQQcKctCT06NZEuXzOH2vNtBwHZoHzm98w83o+Huvh7/L9/X7vNWs58+H7/V0iM5EkSVJ5epQdQJIkqbuzIJMkSSqZBZkkSVLJLMgkSZJKZkEmSZJUMgsySZKkklmQSZIklcyCTJIkqWQWZJIkSSXbruwAW2qPPfbIAQMGlB1DkiSpTYsWLfpzZvZpq902V5ANGDCAhQsXlh1DkiSpTRHxdD3tHLKUJEkqmQWZJElSySzIJEmSSrbNnUPWmnXr1tHY2MjatWvLjqKC9OrVi379+tGzZ8+yo0iS1O66REHW2NjIzjvvzIABA4iIsuOonWUma9asobGxkYEDB5YdR5KkdtclhizXrl3L7rvvbjHWRUUEu+++uz2gkqQuq9CCLCImRMSyiFgREZ9rZf3XI2Jx9fVkRPy/t3GstxdWnZqfrySpKytsyDIiGoBZwPuBRmBBRMzLzCVNbTLzszXt/w44vKg8kiRJnVWRPWSjgBWZuTIz3wBmA8dvpv2pwI3tdfDevXtvdv2AAQMYMmQIQ4cOZciQIfzoRz9qr0MX5rvf/S5Tp05907Jx48Z5o1xJkrZxRZ7Uvw/wTM18IzC6tYYR0R8YCNy1ifXnAOcA7Lfffu0W8O6772aPPfZg2bJlHHvssRx//Obqxb/KTDKTHj3+Ws9u2LCBhoaGdsvW0vr16wvbd2e0fv16ttuuS1xzIklSm4rsIWvtpJ/cRNvJwNzM3NDaysy8MjNHZOaIPn3afBzUmzz33HOMHTuWoUOHMnjwYO677763tHn55ZfZbbfdmue/9rWvMXjwYAYPHszMmTMBWLVqFQcffDDnnXcew4YN45lnnqF3795Mnz6d0aNH88ADD3DnnXdy+OGHM2TIEM444wxef/11HnroIU488UQAfvSjH7HjjjvyxhtvsHbtWvbff38AnnrqKSZMmMDw4cMZM2YMTzzxBABTpkzhggsuYPz48Vx00UVtvtcbb7yRIUOGMHjw4De17927NxdddBHDhw/nmGOO4aGHHmLcuHHsv//+zJs3D6gUlNOmTWPkyJEceuihfPvb32762TNt2jQGDx7MkCFDmDNnDgDnnXde87aTJk3ijDPOAODqq6/mC1/4QvPP6+yzz+Y973kPxx57LK+99lq7vl9JkrqKIrsgGoF9a+b7Ac9uou1k4FNFhLjhhhs47rjj+Id/+Ac2bNjAX/7yl+Z148ePJzNZuXIlN910EwCLFi3i2muv5cEHHyQzGT16NEcddRS77bYby5Yt49prr+Xf/u3fAHj11VcZPHgwM2bMYO3atQwaNIg777yTgw46iNNOO43LL7+cqVOn8sgjjwBw3333MXjwYBYsWMD69esZPbrSYXjOOedwxRVXMGjQIB588EHOO+887rqr0ln45JNPcscdd9DQ0MB3v/td5syZw/3339/8HlasWAHAs88+y0UXXcSiRYvYbbfdOPbYY7n11ls54YQTePXVVxk3bhyXXnopkyZN4gtf+AK33347S5Ys4fTTT2fixIlcffXV7LLLLixYsIDXX3+dI488kmOPPZaHH36YxYsX8+ijj/LnP/+ZkSNHMnbsWMaOHct9993HxIkT+cMf/sBzzz0HwP3338/kyZMBWL58OTfeeCPf+c53+PCHP8wPf/hDPvaxj9X9fiVJejuGT7tuq7ZbdNlp7ZykbUUWZAuAQRExEPgDlaLrIy0bRcS7gN2AB4oIMXLkSM444wzWrVvHCSecwNChQ5vXNQ1ZPvXUUxx99NGMGzeO+++/n0mTJrHTTjsBcOKJJzYXHv379+e9731v8/YNDQ2cdNJJACxbtoyBAwdy0EEHAXD66acza9YsPvOZz3DggQeydOlSHnroIS644ALuvfdeNmzYwJgxY3jllVf41a9+xcknn9y839dff715+uSTT35TcXLKKafwrW99q3l+3LhxACxYsIBx48bR1IP40Y9+lHvvvZcTTjiB7bffngkTJgAwZMgQdthhB3r27MmQIUNYtWoVAD//+c/5zW9+w9y5cwF46aWXWL58Offffz+nnnoqDQ0N9O3bl6OOOooFCxYwZswYZs6cyZIlSzjkkEN48cUXee6553jggQf45je/yZo1axg4cGDzz3v48OGsWrVqi9+vJEndQWEFWWauj4ipwG1AA3BNZj4eETOAhZk5r9r0VGB2Zm5qOPNtGTt2LPfeey8/+clP+PjHP860adM47bQ3V74HHHAAffv2ZcmSJWwuRlOR1qRXr17NxcPmthszZgw//elP6dmzJ8cccwxTpkxhw4YNfPWrX2Xjxo3suuuuLF68uK5jbsrmjt+zZ8/m20b06NGDHXbYoXm66dy0zORf//VfOe6449607fz581vd5z777MOLL77Iz372M8aOHcsLL7zATTfdRO/evdl5551Zs2ZN83GgUry+9tpr7fZ+JUnqSgq9D1lmzs/MgzLzgMz8UnXZ9JpijMy8ODPfco+y9vL000+z5557cvbZZ3PmmWfy8MMPv6XN888/z+9+9zv69+/P2LFjufXWW/nLX/7Cq6++yi233MKYMWPaPM673/1uVq1a1TyE+P3vf5+jjjoKqBSFM2fO5IgjjqBPnz6sWbOGJ554gve85z284x3vYODAgdx8881ApTB69NFHt/h9jh49ml/84hf8+c9/ZsOGDdx4443Nx6/Hcccdx+WXX866deuAytDhq6++ytixY5kzZw4bNmxg9erV3HvvvYwaNQqAI444gpkzZzJ27FjGjBnDV7/61TZ/Vu31fiVJ6kq6/GVs99xzD5dddhk9e/akd+/eXHfdX8eTx48fT0NDA+vWreOSSy6hb9++9O3blylTpjQXHWeddRaHH35489DepvTq1Ytrr72Wk08+mfXr1zNy5EjOPfdcoFIs/elPf2Ls2LEAHHrooey5557NvVbXX389n/zkJ/niF7/IunXrmDx5MocddtgWvc+9996bL3/5y83nxX3wgx+s+6rRpve5atUqhg0bRmbSp08fbr31ViZNmsQDDzzAYYcdRkTwla98hb322guo9Pz9/Oc/58ADD6R///688MILdRWv7fF+JUnqSqKgkcLCjBgxIlved2vp0qUcfPDBJSVSR/FzliRtic5wUn9ELMrMEW216xLPspQkSdqWWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsm6/H3IOtLWXl67KfVcdtvQ0MCQIUOa52+99VYGDBjQattVq1bxoQ99iMcee6y9IkqSpHZgQbaN23HHHTf5GCJJkrRtcMiyC1q1ahVjxoxh2LBhDBs2jF/96ldvafP4448zatQohg4dyqGHHsry5csB+MEPftC8/BOf+AQbNmzo6PiSJHU7FmTbuNdee42hQ4cydOhQJk2aBMCee+7J7bffzsMPP8ycOXM4//zz37LdFVdcwac//WkWL17MwoUL6devH0uXLmXOnDn88pe/ZPHixTQ0NHD99dd39FuSJKnbcchyG9fakOW6deuYOnVqc1H15JNPvmW7I444gi996Us0NjZy4oknMmjQIO68804WLVrEyJEjgUqxt+eee3bI+5AkqTuzIOuCvv71r9O3b18effRRNm7cSK9evd7S5iMf+QijR4/mJz/5CccddxxXXXUVmcnpp5/Ol7/85RJSS5LUfTlk2QW99NJL7L333vTo0YPvf//7rZ4HtnLlSvbff3/OP/98Jk6cyG9+8xuOPvpo5s6dy/PPPw/ACy+8wNNPP93R8SVJ6nbsIWtH7fl0+LfjvPPO46STTuLmm29m/Pjx7LTTTm9pM2fOHH7wgx/Qs2dP9tprL6ZPn8473/lOvvjFL3LssceyceNGevbsyaxZs+jfv38J70KSpO4jMrPsDFtkxIgRuXDhwjctW7p0KQcffHBJidRR/JwlSVtia+8P2p4dLBGxKDNHtNXOIUtJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsm8D1k7+v2MIe26v/2m/3az69esWcPRRx8NwB//+EcaGhro06cPAA899BDbb799u+aRJEnFsCDbhu2+++7Nz7G8+OKL6d27NxdeeOGb2mQmmUmPHnaGSpLUWflXugtasWIFgwcP5txzz2XYsGE888wz7Lrrrs3rZ8+ezVlnnQXAn/70J0488URGjBjBqFGj+PWvf11WbEmSui0Lsi5qyZIlnHnmmTzyyCPss88+m2x3/vnn8/d///csXLiQm266qblQkyRJHcchyy7qgAMOYOTIkW22u+OOO1i2bFnz/Isvvshrr73GjjvuWGQ8SZJUo9CCLCImAN8AGoCrMvOSVtp8GLgYSODRzPxIkZm6i9oHivfo0YPaZ5auXbu2eTozvQBAkqSSFTZkGRENwCzgA8AhwKkRcUiLNoOAzwNHZuZ7gM8Ulac769GjB7vtthvLly9n48aN3HLLLc3rjjnmGGbNmtU833SRgCRJ6jhF9pCNAlZk5kqAiJgNHA8sqWlzNjArM18EyMznC8xTuLZuU1GmSy+9lAkTJrDffvtxyCGH8PrrrwMwa9YsPvnJT3Lttdeyfv16xo8f/6YCTZIkFa/Igmwf4Jma+UZgdIs2BwFExC+pDGtenJk/a7mjiDgHOAdgv/32KyTstu7iiy9unj7wwAPf0tN1yimncMopp7xluz59+jB37tyi40mSpM0o8irLaGVZtpjfDhgEjANOBa6KiF3fslHmlZk5IjNHNN34VJIkqasosiBrBPatme8HPNtKmx9l5rrM/B2wjEqBJkmS1G0UWZAtAAZFxMCI2B6YDMxr0eZWYDxAROxBZQhz5dYcrPYqQnU9fr6SpK6ssIIsM9cDU4HbgKXATZn5eETMiIiJ1Wa3AWsiYglwNzAtM9ds6bF69erFmjVr/KPdRWUma9asoVevXmVHkSSpEIXehywz5wPzWyybXjOdwAXV11br168fjY2NrF69+u3sRp1Yr1696NevX9kxJEkqRJe4U3/Pnj0ZOHBg2TEkSZK2is+ylCRJKpkFmSRJUsksyCRJkkpmQSZJklQyCzJJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsksyCRJkkpmQSZJklQyCzJJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsksyCRJkkpmQSZJklQyCzJJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsksyCRJkkpWaEEWERMiYllErIiIz7WyfkpErI6IxdXXWUXmkSRJ6oy2K2rHEdEAzALeDzQCCyJiXmYuadF0TmZOLSqHJElSZ1dkD9koYEVmrszMN4DZwPEFHk+SJGmbVGRBtg/wTM18Y3VZSydFxG8iYm5E7FtgHkmSpE6pyIIsWlmWLeb/AxiQmYcCdwDfa3VHEedExMKIWLh69ep2jilJklSuws4ho9IjVtvj1Q94trZBZq6pmf0OcGlrO8rMK4ErAUaMGNGyqJMkSSUaPu26rd520WWntWOSbVeRPWQLgEERMTAitgcmA/NqG0TE3jWzE4GlBeaRJEnqlArrIcvM9RExFbgNaACuyczHI2IGsDAz5wHnR8REYD3wAjClqDySJEmdVZFDlmTmfGB+i2XTa6Y/D3y+yAySJEmdnXfqlyRJKpkFmSRJUsksyCRJkkpmQSZJklQyCzJJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsksyCRJkkpmQSZJklQyCzJJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsksyCRJkkpmQSZJklQyCzJJkqSSWZBJkiSVbLt6GkXEQcA0oH/tNpn5voJySZIkdRt1FWTAzcAVwHeADcXFkSRJ6n7qLcjWZ+blhSaRJEnqpuo9h+w/IuK8iNg7It7Z9Co0mSRJUjdRbw/Z6dX/TqtZlsD+7RtHkiSp+6mrIMvMgUUHkSRJ6q7qvcqyJ/BJYGx10T3AtzNzXUG5JEmSuo16zyG7HBgO/Fv1Nby6bLMiYkJELIuIFRHxuc20+9uIyIgYUWceSZKkLqPec8hGZuZhNfN3RcSjm9sgIhqAWcD7gUZgQUTMy8wlLdrtDJwPPFh/bEmSpK6j3h6yDRFxQNNMROxP2/cjGwWsyMyVmfkGMBs4vpV2/wR8BVhbZxZJkqQupd6CbBpwd0TcExG/AO4C/lcb2+wDPFMz31hd1iwiDgf2zcwfb25HEXFORCyMiIWrV6+uM7IkSdK2od6rLO+MiEHAu4AAnsjM19vYLFrbVfPKiB7A14EpdRz/SuBKgBEjRmQbzSVJkrYpmy3IIuJ9mXlXRJzYYtUBEUFm/vtmNm8E9q2Z7wc8WzO/MzAYuCciAPYC5kXExMxcWPc7kCRJ2sa11UN2FJXhyf/RyroENleQLQAGRcRA4A/AZOAjzRtnvgTs0TQfEfcAF1qMSZKk7mazBVlm/t/q5IzM/F3tumqhtblt10fEVOA2oAG4JjMfj4gZwMLMnPc2ckuSJHUZ9d724ofAsBbL5lK5H9kmZeZ8YH6LZdM30XZcnVkkSZK6lLbOIXs38B5glxbnkb0D6FVkMEmSpO6irR6ydwEfAnblzeeR/SdwdlGhJEmSupO2ziH7UUT8GLgoM/+5gzJJkiR1K23eGDYzN1B5/JEkSZIKUO9J/b+KiG8Bc4BXmxZm5sOFpJIkSepG6i3I/mv1vzNqliXwvvaNI0mS1P3U++ik8UUHkSRJ6q7qerh4ROwSEV9resB3RPxLROxSdDhJkqTuoK6CDLiGyq0uPlx9vQxcW1QoSZKk7qTec8gOyMyTaub/MSIWFxFIkiSpu6m3h+y1iPhvTTMRcSTwWjGRJEmSupd6e8g+CXyvet5YAC8ApxeWSpIktWr4tOu2ettFl53WjknUnuq9ynIxcFhEvKM6/3KhqSRJkrqReq+y3D0ivgncA9wdEd+IiN0LTSZJktRN1HsO2WxgNXAS8LfV6TlFhZIkSepO6j2H7J2Z+U8181+MiBOKCCRJktTd1NtDdndETI6IHtXXh4GfFBlMkiSpu6i3IPsEcAPwRvU1G7ggIv4zIjzBX5Ik6W2o9yrLnYsOIkmS1F3Vew4ZETERGFudvSczf1xMJEmSpO6l3tteXAJ8GlhSfX26ukySJElvU709ZB8EhmbmRoCI+B7wCPC5ooJJkiR1F/We1A+wa830Lu0dRJIkqbuqt4fsy8AjEXE3lWdZjgU+X1gqSZKkbqTNgiwiArgfeC8wkkpBdlFm/rHgbJIkSd1CmwVZZmZE3JqZw4F5HZBJkiSpW6n3HLJfR8TIQpNIkiR1U/WeQzYeODciVgGvUhm2zMw8tKhgkiRJ3UW9BdkHtmbnETEB+AbQAFyVmZe0WH8u8ClgA/AKcE5mLtmaY0mSJG2rNluQRUQv4FzgQOC3wNWZub6eHUdEAzALeD/QCCyIiHktCq4bMvOKavuJwNeACVv8LiRJkrZhbZ1D9j1gBJVi7APAv2zBvkcBKzJzZWY2PZD8+NoGmVn7YPKdgNyC/UuSJHUJbQ1ZHpKZQwAi4mrgoS3Y9z7AMzXzjcDolo0i4lPABcD2wPta21FEnAOcA7DffvttQQRJkqTOr60esnVNE/UOVdaIVpa9pQcsM2dl5gHARcAXWttRZl6ZmSMyc0SfPn22MIYkSVLn1lYP2WER0TSsGMCO1fmmqyzfsZltG4F9a+b7Ac9upv1s4PI28kiSJHU5my3IMrPhbex7ATAoIgYCfwAmAx+pbRARgzJzeXX2vwPLkSRJ6mbqve3FFsvM9RExFbiNym0vrsnMxyNiBrAwM+cBUyPiGCpDoy8CpxeVR5IkqbMqrCADyMz5wPwWy6bXTH+6yONLkiRtC+p9dJIkSZIKYkEmSZJUMgsySZKkklmQSZIklcyCTJIkqWQWZJIkSSWzIJMkSSqZBZkkSVLJLMgkSZJKZkEmSZJUskIfnSRJ0rZs+LTrtnrbRZed1o5J1NXZQyZJklQyCzJJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsksyCRJkkpmQSZJklQyCzJJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsksyCRJkkq2XdkBJEl6u34/Y8hWb7vf9N+2YxJp6xRakEXEBOAbQANwVWZe0mL9BcBZwHpgNXBGZj5dZCZJ0taz8JGKUdiQZUQ0ALOADwCHAKdGxCEtmj0CjMjMQ4G5wFeKyiNJktRZFXkO2ShgRWauzMw3gNnA8bUNMvPuzPxLdfbXQL8C80iSJHVKRRZk+wDP1Mw3VpdtypnAT1tbERHnRMTCiFi4evXqdowoSZJUviILsmhlWbbaMOJjwAjgstbWZ+aVmTkiM0f06dOnHSNKkiSVr8iT+huBfWvm+wHPtmwUEccA/wAclZmvF5hHkiSpUyqyh2wBMCgiBkbE9sBkYF5tg4g4HPg2MDEzny8wiyRJUqdVWEGWmeuBqcBtwFLgpsx8PCJmRMTEarPLgN7AzRGxOCLmbWJ3kiRJXVah9yHLzPnA/BbLptdMH1Pk8SVJkrYFPjpJkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsksyCRJkkpmQSZJklSyQm97IUnbit/PGLLV2+43/bftmERSd2QPmSRJUsksyCRJkkpmQSZJklQyCzJJkqSSWZBJkiSVzIJMkiSpZN72QpLUKQyfdt1Wb3vLzu0YRCqBPWSSJEklsyCTJEkqmQWZJElSySzIJEmSSuZJ/ZIkFaAzPh+1M2ZShT1kkiRJJbMgkyRJKplDlpLUSTm8JHUf9pBJkiSVzIJMkiSpZBZkkiRJJbMgkyRJKpkFmSRJUsm8ylKSJJXGq4krCi3IImIC8A2gAbgqMy9psX4sMBM4FJicmXO39BjDp1231fkWXXbaVm8rSduyrf3decvO7RxEElDgkGVENACzgA8AhwCnRsQhLZr9HpgC3FBUDkmSpM6uyB6yUcCKzFwJEBGzgeOBJU0NMnNVdd3GAnNIkiR1akWe1L8P8EzNfGN1mSRJkmoUWZBFK8tyq3YUcU5ELIyIhatXr36bsSRJkjqXIocsG4F9a+b7Ac9uzY4y80rgSoARI0ZsVVHXGq/skCRJnUGRPWQLgEERMTAitgcmA/MKPJ4kSdI2qbCCLDPXA1OB24ClwE2Z+XhEzIiIiQARMTIiGoGTgW9HxONF5ZEkSeqsCr0PWWbOB+a3WDa9ZnoBlaFMSZKkbstHJ0mSJJXMgkySJKlkFmSSJEkl8+HiBfD5mpIkaUtYkEnqMt7OP4Z8aLakMjlkKUmSVDJ7yLoJh1ElSeq87CGTJEkqmQWZJElSySzIJEmSSuY5ZJ3M72cM2ept95v+23ZMIkmSOoo9ZJIkSSWzIJMkSSqZQ5Zqk8OokiQVyx4ySZKkklmQSZIklcyCTJIkqWQWZJIkSSWzIJMkSSqZBZkkSVLJLMgkSZJKZkEmSZJUMgsySZKkknmnfpVm+LTrtnrbW3a+bKu39ekB6khv73vejkEkdWoWZFILW/sHdNFlp7VzEklSd2FBJrWTzvrMz86aS5L0V55DJkmSVDJ7yCR1OHvtJOnNCi3IImIC8A2gAbgqMy9psX4H4DpgOLAGOCUzVxWZSdoWeWK4JHVthRVkEdEAzALeDzQCCyJiXmYuqWl2JvBiZh4YEZOBS4FTisokSZLUljJ68YvsIRsFrMjMlQARMRs4HqgtyI4HLq5OzwW+FRGRmVlgLkntwF47SWo/RZ7Uvw/wTM18Y3VZq20ycz3wErB7gZkkSZI6nSiqMyoiTgaOy8yzqvMfB0Zl5t/VtHm82qaxOv9Utc2aFvs6BzinOvsuYFk7xdwD+HM77au9mKk+ZqpfZ8xlpvqYqX6dMZeZ6tPVM/XPzD5tNSpyyLIR2Ldmvh/w7CbaNEbEdsAuwAstd5SZVwJXtnfAiFiYmSPae79vh5nqY6b6dcZcZqqPmerXGXOZqT5mqihyyHIBMCgiBkbE9sBkYF6LNvOA06vTfwvc5fljkiSpuymshywz10fEVOA2Kre9uCYzH4+IGcDCzJwHXA18PyJWUOkZm1xUHkmSpM6q0PuQZeZ8YH6LZdNrptcCJxeZoQ3tPgzaDsxUHzPVrzPmMlN9zFS/zpjLTPUxEwWe1C9JkqT6+CxLSZKkklmQSZIklazLF2QRcU1EPB8Rj21ifUTENyNiRUT8JiKGdUCmfSPi7ohYGhGPR8Sny84VEb0i4qGIeLSa6R9babNDRMypZnowIgYUmanmuA0R8UhE/LgzZIqIVRHx24hYHBELW1lfxndq14iYGxFPVL9XR3SCTO+q/oyaXi9HxGfKzrWJrJMiIiPi3WUcf3MZIuKzEbE2InYpK1s1x4bq5/hoRDwcEf+1zDxNImKviJgdEU9FxJKImB8RB5WYp+nn9Hj1Z3VBRJT+t7YmV9PrcyVkeKXF/JSI+FZ1+uKI+EM125KIOLWDMmVE/EvN/IURcXHN/GkR8Vj181wSERcWFiYzu/QLGAsMAx7bxPoPAj8FAngv8GAHZNobGFad3hl4EjikzFzV4/SuTvcEHgTe26LNecAV1enJwJwO+gwvAG4AftzKug7PBKwC9tjM+jK+U98DzqpObw/sWnamFsdvAP5I5QaJnSZXTY6bgPuAi8s4/uYyAA9Vl08pK1s1xys108cBvygzTzVHAA8A59YsGwqM6SQ/pz2BO4B/7AQ/q1c6WwZgCvCt6vTFwIXV6UHAy0DPDsi0Fvhd0+904MKm/weBDwAPA39Tne8FnF1UltKr9qJl5r20crPZGscD12XFr4FdI2I0adLbAAAGr0lEQVTvgjM9l5kPV6f/E1jKWx8r1aG5qsdp+tdLz+qr5RUfx1P5ww+VZ48eHRFRVCaAiOgH/Hfgqk006fBMdejQzy4i3kHlHx5XA2TmG5n5/8rM1Iqjgacy8+lOlouI6A0cCZxJSbfe2VSGiDgA6A18AeiQHoM6vQN4sewQwHhgXWZe0bQgMxdn5n0lZmqWmc9TecrM1E7we2mbkZnLgb8Au3XA4dZTuaLys62s+zyVIvHZaq61mfmdooJ0+YKsDvU8c7Mw1SG2w6n0SNXq8FzVocHFwPPA7Zm5yUzZcc8enQn8PbBxE+vLyJTAzyNiUVQe67XJTFVFf3b7A6uBa6tDu1dFxE4lZ2ppMnBjK8vLzgVwAvCzzHwSeKGkYdNNZTiVys/tPuBdEbFnCdma7FgdTnqCyj+Q/qnELE0GA4vKDrE5mbmSyt/aMj87+Ovn1/Q6pewMwIzWGlW//8urBW1HmAV8tJXTAjr0+2VBVunybqlD7gVS/VfxD4HPZObLLVe3skmhuTJzQ2YOpfKYq1ERMbjMTBHxIeD5zNzc/xBlfH5HZuYwKt3Zn4qIsSVn2o7KsPzlmXk48CrQ8vyQMr/n2wMTgZtbW93Kso6+F8+pwOzq9GzK6YnaVIbJwOzM3Aj8O+Xet/G1zByame8GJgDX2etTt87wc2r6/Jpec8rOAExvsf6zEbGMSgfFxR0Vqvr39zrg/I46ZmssyOp75ma7i4ieVIqx6zPz3ztLLoDqcNc9VH7ptpopNvPs0XZ0JDAxIlZR+SP1voj4QcmZqOm+fh64BRi1qUxVRX92jUBjTY/mXCoFWpmZan0AeDgz/9TKujJzERG7A+8Drqp+z6YBp3RkobGZDIdROZfm9uryyXSSYcvMfIDKw5fbfGBywR4HhpecYbMiYn9gA5WRB23e1zPzXcApVAr+Xh147JlUThmoHV3o0O+XBVnleZqnVa/2ei/wUmY+V+QBq7/srwaWZubXOkOuiOgTEbtWp3cEjgGeaCVThz17NDM/n5n9MnMAlT9Gd2Xmx8rMFBE7RcTOTdPAsUDLK3g79LPLzD8Cz0TEu6qLjgaWlJmphaZht9aUmQsq35nrMrN/Zg7IzH2pnOD73zpBhplUTi4eUH39DbBPRPTvwGytisqVoA3AmpKj3AXsEBFnNy2IiJERcVSJmZpFRB/gCionrnsX9jpVOykW8tff7R1xzBeoXFhzZs3iLwNfiYi9oPmq/sJ60Qp9dFJnEBE3AuOAPSKiEfi/VE5Yp3oi6HwqV3qtoHIS4f/sgFhHAh8HflsdRwf438B+JebaG/heRDRQKdRvyswfRyd89mjJmfoCt1Q7ULYDbsjMn0XEuVDqd+rvgOurw4Mrgf/ZCTIREf8FeD/wiZplpeeqcSpwSYtlPwQ+QuW8rTIzfJZKD2ytW6h8xy/tgFwt7Vjz+yqA0zNzQwk5mmVmRsQkYGZUbuOwlspV0J/Z7IbFavo59aRywvj3gU39w7sj1X5+UDlnscNvfbEFZgA3RMR3qkP2HeFfgKlNM5k5PyL6AndUO1ISuKaog/voJEmSpJI5ZClJklQyCzJJkqSSWZBJkiSVzIJMkiSpZBZkkiRJJbMgk9RlRMSG6mNZHouI/2i6t95m2u8aEefVzP9NRMwtPqkkvZm3vZDUZUTEK5nZuzr9PeDJzPzSZtoPAH6cmS0fEyZJHcoeMkld1QNUH1QeEb0j4s6IeDgifhsRx1fbXAIcUO1VuywiBkTEY9VtpkTEv0fEzyJieUR8pWnHEXFmRDwZEfdExHci4lsd/u4kdSld/k79krqf6hMnjqbyJAeo3MF9Uma+HBF7AL+OiHlUHsI+uPqg46Yes1pDgcOB14FlEfGvVJ5L+H+oPC/0P6k8vufRQt+QpC7PgkxSV9L0eJgBwCLg9uryAP45IsYCG6n0nPWtY393ZuZLABGxBOhP5aHav6g++46IuBk4qD3fhKTuxyFLSV3Ja9Xerv7A9sCnqss/CvQBhlfX/wnoVcf+Xq+Z3kDlH7HRfnElqcKCTFKXU+3VOh+4MCJ6ArsAz2fmuogYT6Vgg8qQ485buPuHgKMiYreI2A44qb1yS+q+LMgkdUmZ+QiVc7smA9cDIyJiIZXesieqbdYAv6zeJuOyOvf7B+CfgQeBO4AlwEvt/w4kdSfe9kKStlBE9M7MV6o9ZLcA12TmLWXnkrTtsodMkrbcxdWLBx4DfgfcWnIeSds4e8gkSZJKZg+ZJElSySzIJEmSSmZBJkmSVDILMkmSpJJZkEmSJJXs/wNENPAETDlQtwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (10,5))\n",
    "\n",
    "score_home_final = sb.barplot(data = multi_visual, x = 'Rating', y = 'Proportion', hue = 'IsBorrowerHomeowner')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3sAAAJQCAYAAAA30X2iAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzs3X2cX1V9L/rPIg+AgOVBsIZJCGEoTUIgkBnUnsNDhBhM7QgINHpuhUtR1HC1RaP0dTXWFG85haNeG1rtKQo+ESS9kniE8GBRULHJICmWIISSQGa0SiGtxQIhw7p/EOYkJJAJzG+S7Hm/X6/fi73WXnvPd8+D5vNaa+9daq0BAACgWXbb0QUAAAAw+IQ9AACABhL2AAAAGkjYAwAAaCBhDwAAoIGEPQAAgAYS9gAAABpI2AMAAGggYQ8AAKCBRu7oArbXa17zmjp+/PgdXQYAAMAOcdddd/1rrfXAbY3b5cLe+PHj093dvaPLAAAA2CFKKQ8PZJxlnAAAAA0k7AEAADSQsAcAANBAu9w9e1vzzDPPpKenJ0899dSOLoUW2WOPPdLW1pZRo0bt6FIAAGCX0Iiw19PTk3322Sfjx49PKWVHl8Mgq7XmscceS09PTw499NAdXQ4AALu4pUuX5oMf/GD6+vpy/vnn5+KLL95s/x//8R/ntttuS5L853/+Z375y1/m3/7t35IkV199dS655JIkycc+9rGcc845Q1v8dmhE2HvqqacEvQYrpeSAAw7Io48+uqNLAQBgF9fX15c5c+bklltuSVtbWzo7O9PV1ZVJkyb1j/nMZz7Tv/2Xf/mXufvuu5Mkjz/+eD75yU+mu7s7pZRMmzYtXV1d2W+//Yb8OgaiMffsCXrN5ucLAMBgWLZsWdrb2zNhwoSMHj06s2fPzuLFi190/DXXXJN3vOMdSZKbbropM2bMyP7775/99tsvM2bMyNKlS4eq9O3WmLAHAACwLb29vRk7dmx/u62tLb29vVsd+/DDD2f16tV505vetN3H7gwaG/b23nvvl9w/fvz4TJkyJVOnTs2UKVNeMs3vLK666qpceOGFm/WddNJJXjIPAAADVGvdou/FVpEtXLgwZ555ZkaMGLHdx+4MGhv2BuK2227LihUrsmjRonzgAx8Y8HG11jz77LOb9fX19Q12eZvZsGFDS8+/sxlu1wsAwNBoa2vL2rVr+9s9PT0ZM2bMVscuXLiwfwnn9h67M2h82Pv5z3+eE044IVOnTs2RRx6ZO+64Y4sxv/rVrza7qfLTn/50jjzyyBx55JH57Gc/myRZs2ZNJk6cmPe///059thjs3bt2uy9996ZN29eXv/61+fOO+/Md77znRxzzDGZMmVKzjvvvDz99NNZtmxZzjjjjCTJ4sWLs+eee2b9+vV56qmnMmHChCTJP//zP+fUU0/NtGnTcvzxx+enP/1pkuTcc8/NRRddlOnTp+ejH/3oNq/1mmuuyZQpU3LkkUduNn7vvffORz/60UybNi2nnHJKli1blpNOOikTJkzIkiVLkjwXVufOnZvOzs4cddRR+cIXvpDkuWA7d+7cHHnkkZkyZUquvfbaJMn73//+/mNPP/30nHfeeUmSK6+8Mh/72Mf6v1/vfve7M3ny5Lz5zW/Ok08+OajXCwAA26uzszOrVq3K6tWrs379+ixcuDBdXV1bjLv//vuzbt26vPGNb+zvmzlzZm6++easW7cu69aty80335yZM2cOZfnbp9a6S32mTZtWX2jlypVb9O2111611lovv/zyeskll9Raa92wYUP91a9+VWut9ZBDDqlHHnlknTx5ct1zzz3rt771rVprrd3d3fXII4+sTzzxRP2P//iPOmnSpPrjH/+4rl69upZS6p133tn/NZLUa6+9ttZa65NPPlnb2trq/fffX2ut9Q/+4A/qZz7zmfrMM8/U8ePH11pr/dCHPlQ7Ojrq97///frd7363zp49u9Za65ve9Kb6wAMP1Fpr/dGPflSnT59ea631nHPOqb/7u79bN2zYUGut9Utf+lJ9zWteU48++uj+z1577VWXL19ee3t769ixY+svf/nL+swzz9Tp06fXb37zm/113nDDDbXWWk877bQ6Y8aMun79+rpixYp69NFH11pr/cIXvlD/7M/+rNZa61NPPVWnTZtWH3roobpo0aJ6yimn1A0bNtR/+Zd/qWPHjq0/+9nP6jXXXFM//OEP11pr7ezsrK9//etrrbWee+65denSpXX16tV1xIgR9e6776611nrWWWfVr3zlK9t1vQP5OQMAwPb69re/XQ8//PA6YcKE/qzw8Y9/vC5evLh/zCc+8Yn60Y9+dItjr7zyynrYYYfVww47rH7xi18cspo3laS7DiA7NeLVCy+ls7Mz5513Xp555pmcdtppmTp1av++2267La95zWvyz//8zzn55JNz0kkn5fvf/35OP/307LXXXkmSM844I3fccUe6urpyyCGH5A1veEP/8SNGjMjb3/72JM8l/0MPPTS/9Vu/lSQ555xzcsUVV+SP/uiP0t7envvuuy/Lli3LRRddlNtvvz19fX05/vjj88QTT+SHP/xhzjrrrP7zPv300/3bZ511Vv8a4ST5/d///SxYsKC/fdJJJyVJli9fnpNOOikHHnhgkuS//bf/lttvvz2nnXZaRo8enVNPPTVJMmXKlOy+++4ZNWpUpkyZkjVr1iRJbr755txzzz1ZtGhRkuTf//3fs2rVqnz/+9/PO97xjowYMSKvfe1rc+KJJ2b58uU5/vjj89nPfjYrV67MpEmTsm7duvz85z/PnXfemc997nN57LHHcuihh/Z/v6dNm5Y1a9Zs9/UCAMBgmzVrVmbNmrVZ3/z58zdr/+mf/ulWjz3vvPP6V7Xt7Bof9k444YTcfvvt+fa3v50/+IM/yNy5c/Oud71rszGHHXZYXvva12blypVbvenyec8HwOftscceL3mz5vOOP/743HjjjRk1alROOeWUnHvuuenr68vll1+eZ599Nvvuu29WrFgxoK/5Yl7q648aNar/xtHddtstu+++e//28/fG1Vrzl3/5l1tMQ99www1bPefBBx+cdevWZenSpTnhhBPy+OOP5xvf+Eb23nvv7LPPPnnsscf6v07yXDB+8sknB+16AQCAl9b4e/YefvjhHHTQQXn3u9+dP/zDP8yPf/zjLcb88pe/zOrVq3PIIYfkhBNOyPXXX5///M//zK9//et885vfzPHHH7/Nr/Pbv/3bWbNmTR588MEkyVe+8pWceOKJSZ4LnJ/97Gfzxje+MQceeGAee+yx/PSnP83kyZPz6le/Ooceemiuu+66JM+Frn/8x3/c7ut8/etfn+9973v513/91/T19eWaa67p//oDMXPmzPz1X/91nnnmmSTJAw88kF//+tc54YQTcu2116avry+PPvpobr/99hx33HFJkje+8Y357Gc/mxNOOCHHH398Lr/88m1+rwbregEAgJfW+Jm97373u7nssssyatSo7L333vnyl7/cv2/69OkZMWJEnnnmmVx66aV57Wtfm9e+9rU599xz+wPN+eefn2OOOaZ/ueOL2WOPPfKlL30pZ511VjZs2JDOzs68973vTfJcEPvFL36RE044IUly1FFH5aCDDuqfbfva176W973vfbnkkkvyzDPPZPbs2Tn66KO36zpf97rX5c///M8zffr01Foza9asvO1tbxvw8eeff37WrFmTY489NrXWHHjggbn++utz+umn584778zRRx+dUkr+4i/+Ir/5m7+Z5LkZy5tvvjnt7e055JBD8vjjjw8oGA/G9QIAAC+tvNTyv51RR0dHfeF75e67775MnDhxB1XEUPFzBgCApJRyV621Y1vjWrqMs5Ryainl/lLKg6WUi7ey/zOllBUbPw+UUv6tlfUAAAAMFy1bxllKGZHkiiQzkvQkWV5KWVJrXfn8mFrrH28y/v9Kckyr6gEAABhOWjmzd1ySB2utD9Va1ydZmOSlbiJ7R5JrWlgPAADAsNHKsHdwkrWbtHs29m2hlHJIkkOT/P2L7H9PKaW7lNL96KOPDnqhAAAATdPKsFe20vdiT4OZnWRRrbVvaztrrX9Ta+2otXY8/9JwAAAAXlwrw15PkrGbtNuS/OxFxs6OJZwAAACDppXv2Vue5PBSyqFJevNcoHvnCweVUo5Isl+SO1tYy5CYNvfL2x60He667F3bHDNixIhMmTKlv3399ddn/PjxWx27Zs2avPWtb80//dM/DVaJAADATqplYa/WuqGUcmGSm5KMSPLFWuu9pZT5SbprrUs2Dn1HkoV1V3vh305izz33zIoVK3Z0GQAAsNMb7MmZ5w1kkmZHaOl79mqtN9Raf6vWelit9VMb++ZtEvRSa/3TWusW7+Dj5VuzZk2OP/74HHvssTn22GPzwx/+cIsx9957b4477rhMnTo1Rx11VFatWpUk+epXv9rff8EFF6Svb6u3UQIAADu5loY9Wu/JJ5/M1KlTM3Xq1Jx++ulJkoMOOii33HJLfvzjH+faa6/NBz7wgS2O+/znP58PfvCDWbFiRbq7u9PW1pb77rsv1157bX7wgx9kxYoVGTFiRL72ta8N9SUBAACDoJX37DEEtraM85lnnsmFF17YH9geeOCBLY574xvfmE996lPp6enJGWeckcMPPzzf+c53ctddd6WzszPJc0HyoIMOGpLrAAAABpew10Cf+cxn8trXvjb/+I//mGeffTZ77LHHFmPe+c535vWvf32+/e1vZ+bMmfnbv/3b1Fpzzjnn5M///M93QNUAAMBgsoyzgf793/89r3vd67LbbrvlK1/5ylbvu3vooYcyYcKEfOADH0hXV1fuueeenHzyyVm0aFF++ctfJkkef/zxPPzww0NdPgAAMAjM7A2ineUpPO9///vz9re/Pdddd12mT5+evfbaa4sx1157bb761a9m1KhR+c3f/M3Mmzcv+++/fy655JK8+c1vzrPPPptRo0bliiuuyCGHHLIDrgIAAHglyq72xoOOjo7a3d29Wd99992XiRMn7qCKGCp+zgAAvBJNefVCKeWuWmvHtsZZxgkAANBAwh4AAEADCXsAAAANJOwBAAA0kLAHAADQQMIeAABAA3nP3iB6ZP6UQT3fuHk/ecn9jz32WE4++eQkyb/8y79kxIgROfDAA5Mky5Yty+jRowe1HgAAYNch7O3CDjjggKxYsSJJ8qd/+qfZe++98+EPf3izMbXW1Fqz224mcQEAYDiRABrowQcfzJFHHpn3vve9OfbYY7N27drsu+++/fsXLlyY888/P0nyi1/8ImeccUY6Ojpy3HHH5Uc/+tGOKhsAABhEwl5DrVy5Mn/4h3+Yu+++OwcffPCLjvvABz6Qj3zkI+nu7s43vvGN/hAIAADs2izjbKjDDjssnZ2d2xx366235v777+9vr1u3Lk8++WT23HPPVpYHAAC0mLDXUHvttVf/9m677ZZaa3/7qaee6t+utXqYCwAANJBlnMPAbrvtlv322y+rVq3Ks88+m29+85v9+0455ZRcccUV/e3nH/gCAADs2szsDaJtvSphR/rv//2/59RTT824ceMyadKkPP3000mSK664Iu973/vypS99KRs2bMj06dM3C38AAMCuqWy6vG9X0NHRUbu7uzfru++++zJx4sQdVBFDxc8ZAIBXYtrcL7fkvHdd9q6WnPfFlFLuqrV2bGucZZwAAAANJOwBAAA0UGPC3q62HJXt4+cLAADbpxFhb4899shjjz0mEDRUrTWPPfZY9thjjx1dCgAA7DIa8TTOtra29PT05NFHH93RpdAie+yxR9ra2nZ0GQAAsMtoRNgbNWpUDj300B1dBgAAwE6jEcs4AQAA2JywBwAA0EDCHgAAQAMJewAAAA0k7AEAADSQsAcAANBAwh4AAEADCXsAAAANJOwBAAA0kLAHAADQQMIeAABAAwl7AAAADSTsAQAANJCwBwAA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EO3t7dx///08+uijXfa3trayePFiDh06xPDhw9m6dSsVFRUAvPLKKzz44IO8+eabXHHFFRw4cIBBgwZdVhyXO4kCwCc/+Uleeumly/q7IiIiIiLFSBO0lLhc1rdra2tj4cKFPPnkkxw7dowXX3yR/v37R5GGiIiIiIhkUbFX4jLXtxswYEDn+naZ6uvrWbJkCZBe32737t24Ozt37uSmm27i5ptvBmD48OH069cv7zmIiIiIiMiFNIyzxOWyvt3x48cxM2bPns0bb7zB/Pnz+eIXv9jrGEphtlERERERkXxTsVficlnfrq2tjR//+MccOHCAwYMHM2vWLKZOncqsWbNCi1dERERERC6NhnGWuN6sbwd0Wd8ukUhw2223MWLECAYPHkx1dTUvv/xyXuMXEREREZHuqdgrcZnr250/f55UKkVNTU2XNh3r2wFd1rebPXs2r7zyCr/5zW9oa2vjRz/6ERMmTIgiDRERERERyaJir8Rlrm83fvx45s6d27m+XUNDAwC1tbW0tLSQTCZZt24da9euBWDYsGE8/PDDVFVVUVlZyS233MIdd9wRZTrd2rFjB+PGjSOZTHbGnqm1tZV58+aRTCaZPn06TU1NADQ1NXHllVdSWVlJZWUln/vc5/IcuYiIiIjI5dM9e5LT+nYLFy5k4cKFocaXi46lJXbt2kUikaCqqoqampouPZCZS0ukUilWrlzJ1q1bARg7diyHDx+OKnwRERERkcumnj0parksLSEiIiIiEmfq2Stxr66eHMpxR9cdCeW4vZXL0hIAp0+fZsqUKQwZMoSvfOUrfPzjH89f8CIiIiIiOVCxJ0Utl6UlrrvuOl599VWGDx/OoUOH+MxnPsOxY8cYMmRIaPGKiIiIiPQVDeOUopbL0hIDBw5k+PDhAEydOpWxY8dy/Pjx/AUvIiIiIpIDFXtS1HJZWuKNN96gvb0dgFOnTnHixAnGjBmT9xxERERERC6HhnFKUctcWqK9vZ2lS5d2Li0xbdo0ampqqK2tZdGiRSSTScrLy0mlUgDs3buXuro6ysrK6NevH08++STl5eURZyQiIiIicmlU7EnRu9ylJe6++27uvvvu0OMTEREREQmDhnGKiIiIiIgUIfXsSVEr9qUlRERERER6op49ERERERGRIqRiT0REREREpAip2BMRERERESlCKvZERERERESKkIo9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OcbTQE8Xr+Mo870J6ftoY738Qi+sBjab2TeDYeNx8wSwvOOJuz9vZh8C/i7obHHg21EFB2C6KCIiIiIiIlJ8NIxTRERERESkCKnYExERERERKUIq9kRERERERIqQij0REREREZEipGJPRERERESkCKnYExGRkmNm7WZ22MyOmtkPOtbrvEj7D5rZf854fr2ZbQ8/UhERkcunpRdERKTkmNlb7n518HgjcNzd//tF2lcAP3T3SfmJUEREJHfq2RMRkVL3j8BIADO72sx2m9nLZnbEzO4M2qwFxga9gV81swozOxr8zr1m9j0z22FmJ8zsLzsObGa1ZnbczF40s2+a2fq8ZyciIiWrLOoAREREomJm/YBZwIZg07vAXe7+ppmNAF4yswbgUWCSu1cGv1eRdahKYArQCvzczP4n0A58GbgF+DdgD/CTUBMSERHJoGJPRERK0ZVmdhioAA4Bu4LtBvy5mX0CeJ90j9+HLuF4u939HICZNQI3ACOAH7n7L4Pt24A/6MskRERELkbDOEVEpBS9E/TS3QAMAJYF2+8BrgGmBvv/FRh0CcdrzXjcTvpiqvVduCIiIr2nYk9EREpW0Bv3p8AjZtYfGAq87u7vmdmnSBeDkB6G+YFeHn4/cJuZDTOzMuDuvopbRETkUqjYExGRkubu/0T6Xrr5wDPANDM7SLqX72dBmxbgH4KlGr56icc9C/w5sA/4O6ARONf3GYiIiHRPSy+IiIiExMyudve3gp6954Bvu/tzUcclIiKlQT17IiIi4VkVTARzFDgNfD/ieEREpISoZ09ERERERKQIqWdPRERERESkCKnYExERERERKUIq9kRERERERIqQij0REREREZEipGJPRERERESkCKnYExERERERKUL/H/bpVhnW65SOAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1080x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (15,10))\n",
    "\n",
    "score_home_final = sb.barplot(data = multi_visual, x = 'Rating', y = 'Proportion', hue = 'IsBorrowerHomeowner')\n",
    "\n",
    "for rect in score_home_final.patches:\n",
    "        height = rect.get_height()\n",
    "        score_home_final.text(rect.get_x() + rect.get_width()/2., 1.02*height,\n",
    "                '{:0.2f}'.format(height),\n",
    "                ha='center', va='bottom')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_loans.to_csv('main.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [],
   "source": [
    "default_off.to_csv('default.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> If we look at the first proportion (11%), this shows that 11% of the borrowers with rating equal to 1 are non-homeowners and defaulted or got loans charged off. \n",
    ">\n",
    "> Around 50% of the high risk borrowers are non-homeowners that defaulted or got their loans charged off. In total, 61% of the HR borrowers, either homeowners or not, are in this interest group. \n",
    ">\n",
    "> As a general trend, we can note that lower rating means a higher proportion of defaulted or charged off loans (21% of the loans with rating equal to 1 and 40% of the loans with rating equal to D defaulted or got charged off). \n",
    ">\n",
    "> Also, generally speaking non-homeowners defaulted more than others as we can see their predominance apart from the very high ratings where there is a much higher percentage of homeowners.\n",
    ">\n",
    "> In summary, it seems that owning a home and credit rating are good indicators of likelihood of default.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Talk about some of the relationships you observed in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?\n",
    "\n",
    ">  During the last phase of this exploratory analysis, it became obvious that owning a home and credit rating are good indicators of loan default (or writing off). For instance, 61% of the HR loans defaulted/ got charged off and the majority of these were non-homeowners.\n",
    "\n",
    "### Were there any interesting or surprising interactions between features?\n",
    "\n",
    "> I expected defaulted/ charged off loans to be closed mostly after the final term date. However, 18% of the overall population of loans is closed after the term final date whereas this is true for only 4% of the defaulted/ charged off loans. This is a sign that loans that are meant to default get identified generally earlier than the final date, which is a good thing for Prosper."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
