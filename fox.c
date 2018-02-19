/**
 * Algoritmo de Fox 
 * 
 * Paralelo com MPI
 * 
 * Para compilar utilize: mpicc caique_ludio.c -lm
 * Para Executar utilize: mpirun -np NUMEROPROCESSADORES ./a.out
 * 
 * Alunos Lúdio e Caique
 * 
 * */
#include <stdio.h>
#include "mpi.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>


/**
 * Struct de GRID
 * p    numero total de processos
 * comm communicator para o gridinteiro
 * row_comm communicator para a linha atual
 * col_comm communicator para a coluna atual
 * linhaAtual 
 * colunaAtual
 * actualRank
 * */

FILE *arquivoLeitura;
 
#define MASTER_RANK 0
typedef struct {
    int       p;       
    MPI_Comm  comm;    
    MPI_Comm  row_comm;
    MPI_Comm  col_comm;
    int       q;     
    int       linhaAtual;  
    int       colunaAtual; 
    int       actualRank;  
} GRID_INFO_T;

#define MAX 65536
typedef struct {
    int     tamanhoAmostra;
    // Helpers
    #define TamanhoAmostra(A) ((A)->tamanhoAmostra)
    int  entradas[MAX];
    
    #define Entrada(A,i,j) (*(((A)->entradas) + ((A)->tamanhoAmostra)*(i) + (j)))
} MATRIZ_LOCAL_STR;

MATRIZ_LOCAL_STR*  AlocaMatrizLocal(int tamanhoAmostra);
void DestroiMatrizLocal(MATRIZ_LOCAL_STR** local_A);
void LeituraMatriz(MATRIZ_LOCAL_STR* local_A, GRID_INFO_T* grid, int n);
void ImprimeMatriz(MATRIZ_LOCAL_STR* local_A, GRID_INFO_T* grid, int n);
void ZeraMatriz(MATRIZ_LOCAL_STR* local_A);
void MultiplicaMatrizesLocais(MATRIZ_LOCAL_STR* local_A, MATRIZ_LOCAL_STR* local_B, MATRIZ_LOCAL_STR* matrizProdutoSaida);
void Build_matrix_type(MATRIZ_LOCAL_STR* local_A);
MPI_Datatype local_matrix_mpi_t;

MATRIZ_LOCAL_STR*  temp_mat;
void ImprimeMatrizLocal(MATRIZ_LOCAL_STR* local_A, GRID_INFO_T* grid);

int main(int argc, char* argv[]) {
    int p, actualRank, n, tamanhoAmostra; 
    GRID_INFO_T grid;
    MATRIZ_LOCAL_STR* local_A;
    MATRIZ_LOCAL_STR* local_B;
    MATRIZ_LOCAL_STR* matrizProdutoSaida;
    
    void IniciaGrid(GRID_INFO_T*  grid);
    void ExecutaAlgoritmoFox(int n, GRID_INFO_T* grid, MATRIZ_LOCAL_STR* local_A, MATRIZ_LOCAL_STR* local_B, MATRIZ_LOCAL_STR* matrizProdutoSaida);

    if (argc < 1) {
        printf("arquivo de leitura nao passado... utilize mpiexec -np NUMEROPROCESSADORES ./ARQUIVOENTRADA\n");
        return -1;
    }
    
    arquivoLeitura = fopen(argv[1], "r");
    
    if (arquivoLeitura == false)
    {
        printf("problema ao ler arquivo \n");
        return -1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &actualRank);

    IniciaGrid(&grid);
    
    // Se for o master rank, pede o tamanho da matriz.
    if (actualRank == MASTER_RANK) {
        printf("Tamanho da Matriz\n");
        fscanf(arquivoLeitura, "%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    tamanhoAmostra = n/grid.q;

    local_A = AlocaMatrizLocal(tamanhoAmostra);
    TamanhoAmostra(local_A) = tamanhoAmostra;
    LeituraMatriz(local_A, &grid, n);
    ImprimeMatriz(local_A, &grid, n);

    local_B = AlocaMatrizLocal(tamanhoAmostra);
    TamanhoAmostra(local_B) = tamanhoAmostra;
    LeituraMatriz(local_B, &grid, n);
    ImprimeMatriz(local_B, &grid, n);

    Build_matrix_type(local_A);
    temp_mat = AlocaMatrizLocal(tamanhoAmostra);

    matrizProdutoSaida = AlocaMatrizLocal(tamanhoAmostra);
    TamanhoAmostra(matrizProdutoSaida) = tamanhoAmostra;
    ExecutaAlgoritmoFox(n, &grid, local_A, local_B, matrizProdutoSaida);

    if (actualRank == MASTER_RANK)
        printf("Matriz Produto\n"); 
    ImprimeMatriz(matrizProdutoSaida, &grid, n);

    DestroiMatrizLocal(&local_A);
    DestroiMatrizLocal(&local_B);
    DestroiMatrizLocal(&matrizProdutoSaida);

    MPI_Finalize();
} 



void IniciaGrid(GRID_INFO_T*  grid) {
    int processadorAnterior, dimensoes[2], wrap_around[2], coordinates[2], coordenadasLivres[2];

    /**
     * Define as informacoes globais da grade
     * */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD, &processadorAnterior);

    /**
     * Assumindo que o nosso P eh um quadrado perfeito,
     * entao, definimos a dimensao MxM do grid
     * */
    grid->q = (int) sqrt(grid->p);
    dimensoes[0] = dimensoes[1] = grid->q;

    /* circular shift na segunda dimensao */
    wrap_around[0] = wrap_around[1] = 1;
    
    /**
     * Como estamos definindo uma topologia cartesiana,
     * utilizamos a funcao MPI_Cart_Create     * 
     * */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensoes, wrap_around, 1, &(grid->comm));
    
    /**
     * Determina o rank do processo chamador 
     * no comunicador
     * **/
    MPI_Comm_rank(grid->comm, &(grid->actualRank));
    
    /**
     * Define as coordenadas dentro da topologia cartesiana
     * */
    MPI_Cart_coords(grid->comm, grid->actualRank, 2, coordinates);
    grid->linhaAtual = coordinates[0];
    grid->colunaAtual = coordinates[1];

    /**
     * define os comunicadores da linha      * 
     * */
    coordenadasLivres[0] = 0; 
    coordenadasLivres[1] = 1;
    MPI_Cart_sub(grid->comm, coordenadasLivres, &(grid->row_comm));

    /***
     * define os comunicadores da coluna
     * */
    coordenadasLivres[0] = 1; 
    coordenadasLivres[1] = 0;
    MPI_Cart_sub(grid->comm, coordenadasLivres, 
        &(grid->col_comm));
} 


/**
 * 
 * Algoritmo de Fox
 * 
 * Funcao para aplicar o algoritmo de fox
 * 
 * */
void ExecutaAlgoritmoFox(int n, GRID_INFO_T* grid, MATRIZ_LOCAL_STR* local_A, MATRIZ_LOCAL_STR* local_B, MATRIZ_LOCAL_STR* matrizProdutoSaida) {
    clock_t tic = clock();
    
    MATRIZ_LOCAL_STR*  temp_A;
    
    // assumindo o tamanho da amostra n/sqrt(p)
    int stage, bcast_root, tamanhoAmostra, origem, dest;
    MPI_Status       status;

    // define tamanho da amostra atual
    tamanhoAmostra = n/grid->q;
    ZeraMatriz(matrizProdutoSaida);

    // Calcula os enderecos para o shift circular de B    
    origem = (grid->linhaAtual + 1) % grid->q;
    dest = (grid->linhaAtual + grid->q - 1) % grid->q;

    // aloca matriz temporaria
    temp_A = AlocaMatrizLocal(tamanhoAmostra);
    
    for (stage = 0; stage < grid->q; stage++) {
        bcast_root = (grid->linhaAtual + stage) % grid->q;
        if (bcast_root == grid->colunaAtual) {
            MPI_Bcast(local_A, 1, local_matrix_mpi_t, bcast_root, grid->row_comm);
            MultiplicaMatrizesLocais(local_A, local_B, matrizProdutoSaida);
        } else {
            MPI_Bcast(temp_A, 1, local_matrix_mpi_t, bcast_root, grid->row_comm);
            MultiplicaMatrizesLocais(temp_A, local_B, matrizProdutoSaida);
        }
        MPI_Sendrecv_replace(local_B, 1, local_matrix_mpi_t, dest, 0, origem, 0, grid->col_comm, &status);
    }
    
    clock_t toc = clock();
    double time_total = (double)(toc - tic);
    printf("Processo %d terminou em %4.10f milisegundos \n", origem, time_total);
}




MATRIZ_LOCAL_STR* AlocaMatrizLocal(int local_TamanhoAmostra) {
    MATRIZ_LOCAL_STR* temp;
  
    temp = (MATRIZ_LOCAL_STR*) malloc(sizeof(MATRIZ_LOCAL_STR));
    return temp;
}

// Deslocar matriz local
void DestroiMatrizLocal(
         MATRIZ_LOCAL_STR** local_A_ptr  /* in/out */) {
    free(*local_A_ptr);
} 

 /**
  * 
  * Leitura e distribuicao da matriz.
  * 
  * para cada linha da matriz original
  *     para cada grid coluna
  *         leia um bloco de tamanhoAmostra ints no processo 0
  *         e envia para o processo apropriado
  * */
void LeituraMatriz( MATRIZ_LOCAL_STR*  local_A, GRID_INFO_T* grid, int n) {

    int        mat_row, mat_col, grid_row, grid_col, dest, coords[2];
    int*     temp;
    MPI_Status status;
    
    if (grid->actualRank == MASTER_RANK) {
        temp = (int*) malloc(TamanhoAmostra(local_A)*sizeof(int));
        fflush(stdout);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/TamanhoAmostra(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == MASTER_RANK) {
                    for (mat_col = 0; mat_col < TamanhoAmostra(local_A); mat_col++)
                        fscanf(arquivoLeitura, "%d", 
                          (local_A->entradas)+mat_row*TamanhoAmostra(local_A)+mat_col);
                } else {
                    for(mat_col = 0; mat_col < TamanhoAmostra(local_A); mat_col++)
                       fscanf(arquivoLeitura, "%d", temp + mat_col);
                    MPI_Send(temp, TamanhoAmostra(local_A), MPI_INT, dest, 0,
                        grid->comm);
                }
            }
        }
        free(temp);
    } else {
        for (mat_row = 0; mat_row < TamanhoAmostra(local_A); mat_row++) 
            MPI_Recv(&Entrada(local_A, mat_row, 0), TamanhoAmostra(local_A), 
                MPI_INT, 0, 0, grid->comm, &status);
    }
                     
} 


/**
 * Funcao para imprimir a matriz
 * */
void ImprimeMatriz(MATRIZ_LOCAL_STR* local_A, GRID_INFO_T* grid, int n) {
    int        mat_row, mat_col, grid_row, grid_col, origem, coords[2];
    int*     temp;
    MPI_Status status;

    // rank 0, imprime as matrizes
    if (grid->actualRank == MASTER_RANK) {
        temp = (int*) malloc(TamanhoAmostra(local_A)*sizeof(int));

        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/TamanhoAmostra(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &origem);
                if (origem == MASTER_RANK) {
                    for(mat_col = 0; mat_col < TamanhoAmostra(local_A); mat_col++)
                        printf("%d ", Entrada(local_A, mat_row, mat_col));
                } else {
                    MPI_Recv(temp, TamanhoAmostra(local_A), MPI_INT, origem, 0,
                        grid->comm, &status);
                    for(mat_col = 0; mat_col < TamanhoAmostra(local_A); mat_col++)
                        printf("%d ", temp[mat_col]);
                }
            }
            printf("\n");
        }
        free(temp);
    } else {
        for (mat_row = 0; mat_row < TamanhoAmostra(local_A); mat_row++) 
            MPI_Send(&Entrada(local_A, mat_row, 0), TamanhoAmostra(local_A), MPI_INT, 0, 0, grid->comm);
    }
                     
} 

void ZeraMatriz( MATRIZ_LOCAL_STR*  local_A) {
    int i, j;
    for (i = 0; i < TamanhoAmostra(local_A); i++)
        for (j = 0; j < TamanhoAmostra(local_A); j++)
            Entrada(local_A,i,j) = 0;
}

void Build_matrix_type(MATRIZ_LOCAL_STR*  local_A) {
    MPI_Datatype  temp_mpi_t;
    int           block_lengths[2];
    MPI_Aint      displacements[2];
    MPI_Datatype  typelist[2];
    MPI_Aint      start_address, address;

    MPI_Type_contiguous(TamanhoAmostra(local_A)*TamanhoAmostra(local_A), MPI_INT, &temp_mpi_t);

    block_lengths[0] = block_lengths[1] = 1;
    typelist[0] = MPI_INT;
    typelist[1] = temp_mpi_t;
    MPI_Address(local_A, &start_address);
    MPI_Address(&(local_A->tamanhoAmostra), &address);
    displacements[0] = address - start_address;
    
    MPI_Address(local_A->entradas, &address);
    displacements[1] = address - start_address;

    MPI_Type_struct(2, block_lengths, displacements, typelist, &local_matrix_mpi_t);
    MPI_Type_commit(&local_matrix_mpi_t); 
} 

void MultiplicaMatrizesLocais(MATRIZ_LOCAL_STR*  local_A, MATRIZ_LOCAL_STR*  local_B,  MATRIZ_LOCAL_STR*  matrizProdutoSaida) {
    int i, j, k;

    for (i = 0; i < TamanhoAmostra(local_A); i++)
        for (j = 0; j < TamanhoAmostra(local_A); j++)
            for (k = 0; k < TamanhoAmostra(local_B); k++)
                Entrada(matrizProdutoSaida,i,j) = Entrada(matrizProdutoSaida,i,j) 
                    + Entrada(local_A,i,k)*Entrada(local_B,k,j);

}  


void ImprimeMatrizLocal(MATRIZ_LOCAL_STR*  local_A, GRID_INFO_T* grid) {

    int         coords[2], i, j, origem;
    MPI_Status  status;

    if (grid->actualRank == MASTER_RANK) {
        printf("Processo (%d) > Linha da Grade = %d, Coluna da Grade = %d\n", grid->actualRank, grid->linhaAtual, grid->colunaAtual);
        for (i = 0; i < TamanhoAmostra(local_A); i++) {
            for (j = 0; j < TamanhoAmostra(local_A); j++)
                printf("%d ", Entrada(local_A,i,j));
            printf("\n");
        }
        for (origem = 1; origem < grid->p; origem++) {
            MPI_Recv(temp_mat, 1, local_matrix_mpi_t, origem, 0, grid->comm, &status);
            MPI_Cart_coords(grid->comm, origem, 2, coords);
            printf("Processo (%d) > Linha da Grade = %d, Coluna da Grade = %d\n", origem, coords[0], coords[1]);
            for (i = 0; i < TamanhoAmostra(temp_mat); i++) {
                for (j = 0; j < TamanhoAmostra(temp_mat); j++)
                    printf("%d ", Entrada(temp_mat,i,j));
                printf("\n");
            }
        }
        fflush(stdout);
    } else {
        MPI_Send(local_A, 1, local_matrix_mpi_t, 0, 0, grid->comm);
    }
} 
