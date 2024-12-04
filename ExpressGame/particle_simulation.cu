#include "particle_simulation.cuh"

__global__ void UpdateParticlesWithMotion(Particle* particles, int numParticles, Obstacle* obstacles,
    int numObstacles, float mouseX, float mouseY, float targetX,
    float targetY, float targetRadius, bool attract,
    float influenceRadius, int* score, float speed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles && particles[idx].active) {
        float nextX = particles[idx].x; // Position prévue en X
        float nextY = particles[idx].y; // Position prévue en Y

        // Influence de la souris (attraction ou répulsion)
        float dxMouse = mouseX - particles[idx].x;
        float dyMouse = mouseY - particles[idx].y;
        float mouseDistance = sqrtf(dxMouse * dxMouse + dyMouse * dyMouse);

        if (mouseDistance < influenceRadius && mouseDistance > 0.1f) {
            float factor = attract ? speed : -speed;
            nextX += factor * dxMouse / mouseDistance; // Déplacement prévu en X
            nextY += factor * dyMouse / mouseDistance; // Déplacement prévu en Y
        }
        else {
            // Mouvement normal
            nextX += particles[idx].dx * speed;
            nextY += particles[idx].dy * speed;
        }

        // Vérifier si la particule touche la cible
        float dxTarget = targetX - nextX;
        float dyTarget = targetY - nextY;
        float targetDistance = sqrtf(dxTarget * dxTarget + dyTarget * dyTarget);

        if (targetDistance < targetRadius) {
            particles[idx].active = false; // Désactiver la particule
            atomicAdd(score, 1);          // Marquer un point
            return;                       // Arrêter le traitement pour cette particule
        }

        // Vérification des collisions avec les obstacles
        bool collision = false;
        for (int i = 0; i < numObstacles; i++) {
            Obstacle obs = obstacles[i];
            if (nextX > obs.x && nextX < obs.x + obs.width &&
                nextY > obs.y && nextY < obs.y + obs.height) {
                collision = true;
                break;
            }
        }

        // Appliquer le mouvement uniquement s'il n'y a pas de collision
        if (!collision) {
            particles[idx].x = nextX;
            particles[idx].y = nextY;
        }
        else {
            // Gérer le rebond en cas de collision
            particles[idx].dx *= -1.0f;
            particles[idx].dy *= -1.0f;
        }

        // Gestion des bords de l'écran
        if (particles[idx].x < 0 || particles[idx].x > 800) particles[idx].dx *= -1.0f;
        if (particles[idx].y < 0 || particles[idx].y > 600) particles[idx].dy *= -1.0f;
    }
}

// Fonction pour initialiser les particules sur le GPU
Particle* InitializeParticlesGPU(int numParticles, int screenWidth, int screenHeight) {
    std::vector<Particle> hostParticles(numParticles);

    // Initialisation des particules sur le CPU
    for (int i = 0; i < numParticles; i++) {
        float angle = (float)(rand() % 360) * DEG2RAD; // Direction aléatoire
        hostParticles[i] = {
            (float)(rand() % screenWidth),
            (float)(rand() % screenHeight),
            cosf(angle) * 0.5f, // Mouvement en X
            sinf(angle) * 0.5f, // Mouvement en Y
            (unsigned char)(rand() % 256),
            (unsigned char)(rand() % 256),
            (unsigned char)(rand() % 256),
            255,
            true
        };
    }

    // Copier les données vers le GPU
    Particle* deviceParticles;
    cudaMalloc(&deviceParticles, numParticles * sizeof(Particle));
    cudaMemcpy(deviceParticles, hostParticles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    return deviceParticles;
}

// Fonction pour mettre à jour les particules
void UpdateParticles(Particle* deviceParticles, int numParticles, Obstacle* deviceObstacles, int numObstacles,
    float mouseX, float mouseY, float targetX, float targetY, float targetRadius, bool attract,
    float influenceRadius, int* deviceScore, float speed) {
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    UpdateParticlesWithMotion << <numBlocks, blockSize >> > (deviceParticles, numParticles, deviceObstacles,
        numObstacles, mouseX, mouseY, targetX, targetY,
        targetRadius, attract, influenceRadius, deviceScore, speed);
}
