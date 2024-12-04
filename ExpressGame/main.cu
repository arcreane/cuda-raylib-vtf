#include "raylib.h"
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include "renderer.hpp"
#include "interaction.hpp"
#include "timer.hpp"
#include "particle_simulation.cuh"




int main() {
    const int screenWidth = 800;
    const int screenHeight = 600;
    const int numParticles = 1000;
    const int numObstacles = 3;
    const float influenceRadius = 150.0f; // Rayon d'influence de la souris
    const float targetRadius = 20.0f;     // Rayon de la cible



    // Initialiser la fen�tre
    InitGameWindow(screenWidth, screenHeight);

    // Initialiser le timer
    Timer timer(30.0f);

    // Sound
    InitAudioDevice();
    Music music = LoadMusicStream("hyper.mp3");
    PlayMusicStream(music);

    // Initialiser les particules sur le GPU
    Particle* deviceParticles = InitializeParticlesGPU(numParticles, screenWidth, screenHeight);

    // Initialiser les obstacles
    Obstacle obstacles[numObstacles] = {
        {200.0f, 150.0f, 100.0f, 50.0f},
        {500.0f, 300.0f, 150.0f, 50.0f},
        {600.0f, 100.0f, 80.0f, 200.0f}
    };

    Obstacle* deviceObstacles;
    cudaMalloc(&deviceObstacles, numObstacles * sizeof(Obstacle));
    cudaMemcpy(deviceObstacles, obstacles, numObstacles * sizeof(Obstacle), cudaMemcpyHostToDevice);

    // Initialisation du score sur le GPU
    int* deviceScore;
    int hostScore = 0;
    cudaMalloc(&deviceScore, sizeof(int));
    cudaMemcpy(deviceScore, &hostScore, sizeof(int), cudaMemcpyHostToDevice);

    // Position de la cible (centr�e au d�but)
    float targetX = screenWidth / 2.0f;
    float targetY = screenHeight / 2.0f;

    // Vitesse initiale
    float speed = 1.0f;

    // D�tection de victoire
    bool victory = false;

    // Boucle principale
    while (!WindowShouldClose() && !victory) {
        float mouseX, mouseY;
        bool attract = false, repel = false;

        // Mettre � jour le timer
        timer.Update();
        

        // Maj lecture musique
        UpdateMusicStream(music);

        // G�rer les entr�es utilisateur (vitesse, position de la souris, etc.)
        ProcessUserInput(speed, mouseX, mouseY, attract, repel);

        // Mise � jour des particules avec CUDA
        int blockSize = 256;
        int numBlocks = (numParticles + blockSize - 1) / blockSize;
        UpdateParticles(deviceParticles, numParticles, deviceObstacles, numObstacles, mouseX, mouseY, targetX,
            targetY, targetRadius, attract, influenceRadius, deviceScore, speed);

        // Copier le score pour v�rifier la victoire
        cudaMemcpy(&hostScore, deviceScore, sizeof(int), cudaMemcpyDeviceToHost);

        // V�rifier la condition de victoire
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

        for (int i = 0; i < numObstacles; i++) {
            DrawRectangle(obstacles[i].x, obstacles[i].y, obstacles[i].width, obstacles[i].height, GRAY);
        }

        // Afficher le score et la vitesse
        DrawText("Mettre les particules dans la cible rouge | Haut/Bas: changer vitesse", 10, 10, 20, GRAY);
        DrawText(TextFormat("Score: %d", hostScore), 10, 70, 20, WHITE);
        DrawText(TextFormat("Speed: %.2f", speed), 10, 40, 20, GRAY);
        DrawText("Cible rouge: attirer les particules | Haut/Bas: changer vitesse", 10, 70, 20, GRAY);
        DrawText(timer.GetTimeLeft().c_str(), 10, 100, 20, WHITE);

        EndDrawing();
    }

    // Affichage de la victoire
    if (victory) {
        while (!WindowShouldClose()) {
            DrawVictoryScreen(screenWidth, screenHeight);
        }
    }

    // Lib�rer la m�moire GPU
    cudaFree(deviceParticles);
    cudaFree(deviceObstacles);
    cudaFree(deviceScore);

    // Unload music stream buffers from RAM
    UnloadMusicStream(music);   

    // Close audio device (music streaming is automatically stopped)
    CloseAudioDevice();       

    CloseWindow();
    return 0;
}
