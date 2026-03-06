#include <iostream>
#include <mpi.h>
#include <vector>

#define vector std::vector

/*
    MPI Send/Recv Tags
	0 - Matrix A Row
	1 - Matrix B Column
	2 - Result Value, Position
	3 - Position
    4 - Finished Tasks
    5 - Task For Worker
*/

template <typename T>
static vector<T> Flat(const vector<vector<T>> v);

template <typename T>
static vector<vector<T>> Grid(const vector<T> v, int n, int k);

static void SendCalculation(int dest, int row, int col, vector<vector<double>> &a, vector<vector<double>> &b, int n, int k, int m);

static void CrossProduct(int n, int k, int m, vector<vector<double>>& a, vector<vector<double>>& b, vector<vector<double>>& c)
{
    int size, rank;
    
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;

    if (rank == 0)
    {
		int totalTasks = n * m;

        int actual_source;

        // Phase 1: Distribute 1 task to each worker
		int distributedTasks = 0;
        int row = 0;
        int col = 0;
        for (row; row < n; row++)
        {
			//Master process distributes rows and columns to worker processes
            for (col; col < m; col++)
			{
                int dest = (((row * m) + col) % (size - 1)) + 1;
				SendCalculation(dest, row, col, a, b, n, k, m);
				distributedTasks++;

                if (distributedTasks % (size - 1) == 0)
                {
                    col++;
                    if (col >= m)
                    {
                        col = 0;
                        row++;
                    }

                    goto phase2;
				}
            }
            col = 0;
        }	
        phase2:
		//Phase 2: Collect results from fast tasks, distributing new tasks as workers become available
		int tasksReceived = 0;
        for (; row < n; row++)
        {
            for (; col < m; col++)
            {
                double result[3];
                MPI_Recv(&result, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
				tasksReceived++;
                actual_source = status.MPI_SOURCE;

                int dest = actual_source;
				SendCalculation(dest, row, col, a, b, n, k, m);

				int x = result[1];
                int y = result[2];
                //Assign result to correct position in result matrix
				c[x][y] = result[0];
            }
            col = 0;
        }

        //After distribution, master process awaits results from remaining processes
        while (tasksReceived < totalTasks)
        {
            //Receive result
            double result[3];

            MPI_Recv(&result, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			tasksReceived++;

			//std::cout << "Received result for [" << result[1] << ", " << result[2] << "] from " << source << std::endl;
            int x = result[1];
            int y = result[2];

            //Assign result to correct position in result matrix
            c[x][y] = result[0];
        }

		for (int i = 1; i < size; i++)
        {
            MPI_Send(NULL, 0, MPI_INT, i, 4, MPI_COMM_WORLD);
        }
    }
    //Worker processes receive rows and columns, compute the dot product, and send back the result
    else
    {
        //Set up for receive
        double* buf = new double[2 + k + k];
        int position[2];
        double* rowA = new double[k];
        double* colB = new double[k];

		while (true)
        {
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == 4)
            {
                MPI_Recv(NULL, 0, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            }

            MPI_Recv(buf, 2 + k + k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            position[0] = (int)buf[0];
			position[1] = (int)buf[1];

			for (int i = 0; i < k; i++)
            {
                rowA[i] = buf[2 + i];
            }

			for (int i = 0; i < k; i++)
            {
                colB[i] = buf[2 + k + i];
            }

            //Calculate sum of products
            double sum = 0;
            for (int i = 0; i < k; i++)
            {
                sum += *(rowA + i) * *(colB + i);
            }

			//std::cout << "Calculated " << sum << " at " << rank << std::endl;

            //Send back result and position
            double result[] = { sum, position[0], position[1]};
            MPI_Send(&result, 3, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }

        delete[] buf;
        delete[] rowA;
        delete[] colB;
    }

    // Synchronize all processes before exiting
    MPI_Barrier(MPI_COMM_WORLD);
}

template <typename T>
static vector<T> Flat(const vector<vector<T>> v)
{
    vector<T> flat;
    for (int row = 0; row < v.size(); row++)
    {
        for (int col = 0; col < v[0].size(); col++)
        {
            flat.push_back(v[row][col]);
        }
    }

    return flat;
}

template <typename T>
static vector<vector<T>> Grid(const vector<T> v, int n, int k)
{
    vector<vector<T>> result(n, vector<T>(k));

    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < k; col++)
        {
            result[row][col] = v[row * k + col];
        }
    }

    return result;
}

static void SendCalculation(int dest, int row, int col, vector<vector<double>> &a, vector<vector<double>> &b, int n, int k, int m)
{
    vector<double> toSend;
    toSend.push_back(row);
	toSend.push_back(col);

	for (int i = 0; i < k; i++)
    {
        toSend.push_back(a[row][i]);
    }

	for (int i = 0; i < k; i++)
    {
        toSend.push_back(b[i][col]);
    }
    
	void* buf = toSend.data();

	MPI_Send(buf, 2 + k + k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}