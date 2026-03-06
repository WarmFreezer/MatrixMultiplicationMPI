#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

#include "Matrix.cpp"

#define cout std::cout
#define cin std::cin
#define string std::string
#define vector std::vector

std::mt19937 rng(std::random_device{}());
std::uniform_int_distribution<int> dist(-10000, 10000);

static bool ValidDouble(const string str);
static bool ValidInt(const string str);
static void PopulateMatrix(vector<vector<double>>& matrix);
static void PrintMatrix(const vector<vector<double>>& matrix);

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n, k, m;

	if (rank == 0)
	{
		cout << "Enter size of n: ";
		string input = "\n";
		while (!ValidInt(input))
		{
			cin >> input;
		}
		n = stoi(input);

		cout << "Enter size of k: ";
		input = "\n";
		while (!ValidInt(input))
		{
			cin >> input;
		}
		k = stoi(input);

		cout << "Enter size of m: ";
		input = "\n";
		while (!ValidInt(input))
		{
			cin >> input;
		}
		m = stoi(input);
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

	vector<vector<double>> a = {};
	vector<vector<double>> b = {};
	vector<vector<double>> c = {};

	std::chrono::steady_clock::time_point start;

	if (rank == 0)
	{
		std::cerr << "Process " << rank << ": Populating matrices and broadcasting dimensions to all processes.\n";

		a = vector<vector<double>>(n, vector<double>(k));
		b = vector<vector<double>>(k, vector<double>(m));
		c = vector<vector<double>>(n, vector<double>(m));

		PopulateMatrix(a);
		PopulateMatrix(b);

		start = std::chrono::high_resolution_clock::now();
	}

	CrossProduct(n, k, m, a, b, c);

	if (rank == 0)
	{
		auto end = std::chrono::high_resolution_clock::now();

		auto elapsed = end - start;
		cout << elapsed.count() << std::endl;

		if (n <= 10 && k <= 10 && m <= 10)
		{
			PrintMatrix(a);
			cout << std::endl;
			PrintMatrix(b);
			cout << std::endl;
			PrintMatrix(c);
		}

		cout << "Successfully calculated cross product of A and B. Result stored in C.\n";
	}

	MPI_Finalize();


	return 0;
}

static void PrintMatrix(const vector<vector<double>>& matrix)
{
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[i].size(); j++)
		{
			cout << matrix[i][j] << " ";
		}
		cout << "\n";
	}
}

static bool ValidDouble(const string str)
{
	for (char c : str)
	{
		if (!isdigit(c) && c != '.' && c != '-') return false;
	}

	return true;
}

static bool ValidInt(const string str)
{
	for (char c : str)
	{
		if (!isdigit(c)) return false;
	}

	return true;
}

static void PopulateMatrix(vector<vector<double>>& matrix)
{
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[i].size(); j++)
		{
			matrix[i][j] = dist(rng);
		}
	}
}