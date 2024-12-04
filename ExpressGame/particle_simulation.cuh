#include "raylib.h"
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// Structure pour une particule
struct Particle {
    float x, y;         // Position
    float dx, dy;       // Direction de déplacement
    unsigned char r, g, b, a; // Couleur
    bool active;        // Statut de la particule (active/inactive)
};

// Structure pour un obstacle
struct Obstacle {
    float x, y;        // Position
    float width, height; // Dimensions
};

#ifndef PARTICLE_SIMULATION_CUH
// Prototypes de fonctions
Particle* InitializeParticlesGPU(int numParticles, int screenWidth, int screenHeight);
void UpdateParticles(Particle* deviceParticles, int numParticles, Obstacle* obstacles, int numObstacles,
    float mouseX, float mouseY, float targetX, float targetY, float targetRadius,
    bool attract, float influenceRadius, int* deviceScore, float speed);
#endif // !PARTICLE_SIMULATION_CUH



