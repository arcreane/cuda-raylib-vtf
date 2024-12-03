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

// Prototypes de fonctions
void InitGameWindow(int screenWidth, int screenHeight);
Particle* InitializeParticlesGPU(int numParticles, int screenWidth, int screenHeight);
void UpdateParticles(Particle* deviceParticles, int numParticles, float mouseX, float mouseY, float targetX,
    float targetY, float targetRadius, bool attract, float influenceRadius,
    int* deviceScore, float speed);
void DrawVictoryScreen(int screenWidth, int screenHeight);
void ProcessUserInput(float& speed, float& mouseX, float& mouseY, bool& attract, bool& repel);

__global__ void UpdateParticlesWithMotion(Particle* particles, int numParticles, float mouseX, float mouseY,
    float targetX, float targetY, float targetRadius, bool attract,
    float influenceRadius, int* score, float speed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles && particles[idx].active) {
        float dxMouse = mouseX - particles[idx].x;
        float dyMouse = mouseY - particles[idx].y;
        float mouseDistance = sqrtf(dxMouse * dxMouse + dyMouse * dyMouse);

        // Influence de la souris
        if (mouseDistance < influenceRadius && mouseDistance > 0.1f) {
            float factor = attract ? speed : -speed; // Attraction ou répulsion
            particles[idx].x += factor * dxMouse / mouseDistance;
            particles[idx].y += factor * dyMouse / mouseDistance;
        }
        else {
            // Mouvement constant
            particles[idx].x += particles[idx].dx;
            particles[idx].y += particles[idx].dy;
        }

        // Vérification si la particule atteint la cible
        float dxTarget = targetX - particles[idx].x;
        float dyTarget = targetY - particles[idx].y;
        float targetDistance = sqrtf(dxTarget * dxTarget + dyTarget * dyTarget);

        if (targetDistance < targetRadius) {
            particles[idx].active = false; // Désactiver la particule
            atomicAdd(score, 1); // Incrémenter le score
        }

        // Gestion des bords
        if (particles[idx].x < 0 || particles[idx].x > 800) particles[idx].dx *= -1.0f;
        if (particles[idx].y < 0 || particles[idx].y > 600) particles[idx].dy *= -1.0f;
    }
}

int main() {
    const int screenWidth = 800;
    const int screenHeight = 600;
    const int numParticles = 1000;
    const float influenceRadius = 150.0f; // Rayon d'influence de la souris
    const float targetRadius = 20.0f;     // Rayon de la cible

    // Initialiser la fenêtre
    InitGameWindow(screenWidth, screenHeight);

    // Initialiser les particules sur le GPU
    Particle* deviceParticles = InitializeParticlesGPU(numParticles, screenWidth, screenHeight);

    // Initialisation du score sur le GPU
    int* deviceScore;
    int hostScore = 0;
    cudaMalloc(&deviceScore, sizeof(int));
    cudaMemcpy(deviceScore, &hostScore, sizeof(int), cudaMemcpyHostToDevice);

    // Position de la cible (centrée au début)
    float targetX = screenWidth / 2.0f;
    float targetY = screenHeight / 2.0f;

    // Vitesse initiale
    float speed = 1.0f;

    // Détection de victoire
    bool victory = false;

    // Boucle principale
    while (!WindowShouldClose() && !victory) {
        float mouseX, mouseY;
        bool attract = false, repel = false;

        // Gérer les entrées utilisateur (vitesse, position de la souris, etc.)
        ProcessUserInput(speed, mouseX, mouseY, attract, repel);

        // Mise à jour des particules avec CUDA
        int blockSize = 256;
        int numBlocks = (numParticles + blockSize - 1) / blockSize;
        UpdateParticles(deviceParticles, numParticles, mouseX, mouseY, targetX, targetY, targetRadius,
            attract, influenceRadius, deviceScore, speed);

        // Copier le score pour vérifier la victoire
        cudaMemcpy(&hostScore, deviceScore, sizeof(int), cudaMemcpyDeviceToHost);

        // Vérifier la condition de victoire
        if (hostScore >= numParticles) {
            victory = true;
        }

        // Affichage
        BeginDrawing();
        ClearBackground(BLACK);

        // Dessiner la cible
        DrawCircle((int)targetX, (int)targetY, targetRadius, RED);

        // Dessiner les particules
        std::vector<Particle> hostParticles(numParticles);
        cudaMemcpy(hostParticles.data(), deviceParticles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
        for (const auto& particle : hostParticles) {
            if (particle.active) {
                DrawCircle((int)particle.x, (int)particle.y, 2.0f, { particle.r, particle.g, particle.b, particle.a });
            }
        }

        // Afficher le score et la vitesse
        DrawText(TextFormat("Score: %d", hostScore), 10, 10, 20, WHITE);
        DrawText(TextFormat("Speed: %.2f", speed), 10, 40, 20, GRAY);
        DrawText("Cible rouge: attirer les particules | Haut/Bas: changer vitesse", 10, 70, 20, GRAY);

        EndDrawing();
    }

    // Affichage de la victoire
    if (victory) {
        DrawVictoryScreen(screenWidth, screenHeight);
    }

    // Libérer la mémoire GPU
    cudaFree(deviceParticles);
    cudaFree(deviceScore);
    CloseWindow();
    return 0;
}

// Fonction pour initialiser la fenêtre et les ressources
void InitGameWindow(int screenWidth, int screenHeight) {
    InitWindow(screenWidth, screenHeight, "Simulation CUDA - Victoire");
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
void UpdateParticles(Particle* deviceParticles, int numParticles, float mouseX, float mouseY, float targetX,
    float targetY, float targetRadius, bool attract, float influenceRadius,
    int* deviceScore, float speed) {
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    UpdateParticlesWithMotion << <numBlocks, blockSize >> > (deviceParticles, numParticles, mouseX, mouseY,
        targetX, targetY, targetRadius, attract,
        influenceRadius, deviceScore, speed);
}

// Fonction pour afficher l'écran de victoire
void DrawVictoryScreen(int screenWidth, int screenHeight) {
    BeginDrawing();
    ClearBackground(BLACK);
    DrawText("VICTOIRE !", screenWidth / 2 - 100, screenHeight / 2 - 20, 40, GREEN);
    DrawText("Appuyez sur Echap pour quitter.", screenWidth / 2 - 150, screenHeight / 2 + 30, 20, WHITE);
    EndDrawing();
}

// Fonction pour gérer les entrées utilisateur
void ProcessUserInput(float& speed, float& mouseX, float& mouseY, bool& attract, bool& repel) {
    // Contrôle de la vitesse
    if (IsKeyDown(KEY_UP)) speed += 0.1f;   // Augmenter la vitesse
    if (IsKeyDown(KEY_DOWN)) speed -= 0.1f; // Réduire la vitesse
    if (speed < 0.1f) speed = 0.1f;         // Vitesse minimale

    // Gérer la souris
    mouseX = GetMouseX();
    mouseY = GetMouseY();
    attract = IsMouseButtonDown(MOUSE_BUTTON_LEFT); // Attraction
    repel = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);  // Répulsion
}

