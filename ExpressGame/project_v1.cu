#include "raylib.h"
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Structure pour une particule
struct Particle {
    float x, y; // Position
    unsigned char r, g, b, a; // Couleur
};

// Kernel CUDA pour mettre à jour les positions des particules
__global__ void UpdateParticles(Particle* particles, int numParticles, int screenWidth, int screenHeight, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        // Initialisation du générateur aléatoire
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Génération aléatoire
        float dx = curand_uniform(&state) * 2.0f - 1.0f; // Valeur entre -1 et 1
        float dy = curand_uniform(&state) * 2.0f - 1.0f;

        particles[idx].x += dx;
        particles[idx].y += dy;

        // Gestion des bords
        if (particles[idx].x < 0) particles[idx].x = 0;
        if (particles[idx].x > screenWidth) particles[idx].x = screenWidth;
        if (particles[idx].y < 0) particles[idx].y = 0;
        if (particles[idx].y > screenHeight) particles[idx].y = screenHeight;
    }
}

// Fonction pour initialiser les particules sur le GPU
Particle* InitializeParticlesGPU(int numParticles, int screenWidth, int screenHeight) {
    std::vector<Particle> hostParticles(numParticles);

    // Initialisation des particules sur le CPU
    for (int i = 0; i < numParticles; i++) {
        hostParticles[i] = { (float)(rand() % screenWidth), (float)(rand() % screenHeight),
                             (unsigned char)(rand() % 256), (unsigned char)(rand() % 256),
                             (unsigned char)(rand() % 256), 255 };
    }

    // Copier les données vers le GPU
    Particle* deviceParticles;
    cudaMalloc(&deviceParticles, numParticles * sizeof(Particle));
    cudaMemcpy(deviceParticles, hostParticles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    return deviceParticles;
}

int main() {
    const int screenWidth = 800;
    const int screenHeight = 600;
    const int numParticles = 1000;

    InitWindow(screenWidth, screenHeight, "Simulation CUDA - Particules");

    // Initialisation des particules sur le GPU
    Particle* deviceParticles = InitializeParticlesGPU(numParticles, screenWidth, screenHeight);

    // Boucle principale
    while (!WindowShouldClose()) {
        // Lancer le kernel CUDA pour mettre à jour les particules
        int blockSize = 256;
        int numBlocks = (numParticles + blockSize - 1) / blockSize;
        unsigned int seed = time(nullptr); // Seed basé sur l'heure actuelle
        UpdateParticles << <numBlocks, blockSize >> > (deviceParticles, numParticles, screenWidth, screenHeight, seed);

        // Copier les données du GPU vers le CPU pour affichage
        std::vector<Particle> hostParticles(numParticles);
        cudaMemcpy(hostParticles.data(), deviceParticles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

        // Dessiner les particules
        BeginDrawing();
        ClearBackground(BLACK);
        for (const auto& particle : hostParticles) {
            DrawCircle((int)particle.x, (int)particle.y, 2.0f, { particle.r, particle.g, particle.b, particle.a });
        }
        EndDrawing();
    }

    // Libérer la mémoire GPU
    cudaFree(deviceParticles);
    CloseWindow();
    return 0;
}
