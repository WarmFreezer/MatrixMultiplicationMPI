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
*/

template <typename T>
static vector<T> Flat(const vector<vector<T>> v);

template <typename T>
static vector<vector<T>> Grid(const vector<T> v, int n, int k);

static void CrossProduct(int n, int k, int m, vector<vector<double>>& a, vector<vector<double>>& b, vector<vector<double>>& c)
{
    int size, rank;
    
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Data type to skip to next row in matrix B
    MPI_Datatype skipMDouble;
    MPI_Type_vector(k, 1, m, MPI_DOUBLE, &skipMDouble);
    MPI_Type_commit(&skipMDouble);

    if (rank == 0)
    {
        //Create buffers for input vectors. Flattened to keep data contiguous.
        vector<double> flatA = Flat(a);
        vector<double> flatB = Flat(b);

        double* bufA = flatA.data();
        double* bufB = flatB.data();

        for (int row = 0; row < n; row++)
        {
			//Master process distributes rows and columns to worker processes
            for (int col = 0; col < m; col++)
            {
				//Determine destination and starting points in buffers
                int dest = (((row * m) + col) % (size - 1)) + 1;
                void* startA = bufA + static_cast<size_t>(k * row);
                void* startB = bufB + static_cast<size_t>(k * col);

                //Send Position
				int position[2] = { row, col };
				MPI_Send(&position, 2, MPI_INT, dest, 3, MPI_COMM_WORLD);

				//Send the necessary data to the worker process
				MPI_Send(startA, k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
				MPI_Send(startB, 1, skipMDouble, dest, 1, MPI_COMM_WORLD);

                //std::cout << "Sent [" << row << ", " << col << "] to " << dest << std::endl;
            }
        }
        //After distribution, master process awaits results from worker processes
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < m; col++)
            {
                int source = (((row * m) + col) % (size - 1)) + 1;

                //Receive result
                double result[3];

                MPI_Recv(&result, 3, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				//std::cout << "Received result for [" << result[1] << ", " << result[2] << "] from " << source << std::endl;

                int x = result[1];
                int y = result[2];

                //Assign result to correct position in result matrix
                c[x][y] = result[0];
            }
        }
    }
    //Worker processes receive rows and columns, compute the dot product, and send back the result
    else
    {
        int totalTasks = n * m;
        int workers = (size - 1);
        
		int baseTasks = totalTasks / workers;
        int remainder = totalTasks % workers;

		int myTasks = baseTasks + (rank <= remainder ? 1 : 0);

		for (int task = 0; task < myTasks; task++)
        {
			//Receive Position
            int position[2];
            MPI_Recv(position, 2, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			//std::cout << "Received [" << position[0] << ", " << position[1] << "] at " << rank << std::endl;

            //Receive data
            double* rowA = new double[k];
            double* colB = new double[k];

            MPI_Recv(rowA, k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(colB, k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			//std::cout << "Received [" << position[0] << ", " << position[1] << "] at " << rank << std::endl;

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

            delete[] rowA;
            delete[] colB;
        }
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