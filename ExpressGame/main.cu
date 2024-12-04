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
    /*
     * Constants
     */
    const int screenWidth = 800;
    const int screenHeight = 600;
    const int numParticles = 1000;        // Number of particles
    const int numObstacles = 3;           // Number of obstacles
    const float influenceRadius = 150.0f; // Mouse influence radius
    const float targetRadius = 25.0f;     // Target radius
    const float duration = 30.0f;

    // Obstacles definition
    Obstacle obstacles[numObstacles] = {
        {200.0f, 150.0f, 100.0f, 50.0f},
        {500.0f, 300.0f, 150.0f, 50.0f},
        {600.0f, 100.0f, 80.0f, 200.0f}
    };

    /*
     * Initialization
     */
    InitGameWindow(screenWidth, screenHeight);
    InitAudioDevice();
    Music music = LoadMusicStream("hyper.mp3");

    SetTargetFPS(60);

    // Main game loop
    bool isTryAgain;

    do {
        isTryAgain = false;
        // Initialize game state
        Timer timer(duration);
        PlayMusicStream(music);

        // CUDA memory allocations
        Particle* deviceParticles = InitializeParticlesGPU(numParticles, screenWidth, screenHeight);
        Obstacle* deviceObstacles = nullptr;
        int* deviceScore = nullptr;
        int hostScore = 0;

        cudaMalloc(&deviceObstacles, numObstacles * sizeof(Obstacle));
        cudaMemcpy(deviceObstacles, obstacles, numObstacles * sizeof(Obstacle), cudaMemcpyHostToDevice);
        cudaMalloc(&deviceScore, sizeof(int));
        cudaMemcpy(deviceScore, &hostScore, sizeof(int), cudaMemcpyHostToDevice);

        // Game variables
        float targetX = screenWidth / 2.0f;
        float targetY = screenHeight / 2.0f;
        float speed = 1.0f;
        bool victory = false;

        // Main game loop
        while (!WindowShouldClose()) {
            float mouseX = 0.0f, mouseY = 0.0f;
            bool attract = false, repel = false;

            // Update logic
            timer.Update();
            UpdateMusicStream(music);
            ProcessUserInput(speed, mouseX, mouseY, attract, repel);

            // CUDA kernel call for particle updates
            int blockSize = 256;
            int numBlocks = (numParticles + blockSize - 1) / blockSize;
            UpdateParticles(deviceParticles, numParticles, deviceObstacles, numObstacles, mouseX, mouseY,
                targetX, targetY, targetRadius, attract, influenceRadius, deviceScore, speed);

            // Copy score from GPU and check victory
            cudaMemcpy(&hostScore, deviceScore, sizeof(int), cudaMemcpyDeviceToHost);
            if (hostScore >= numParticles) {
                victory = true;
                break;
            }

            // Rendering
            BeginDrawing();
            ClearBackground(BLACK);

            DrawCircle((int)targetX, (int)targetY, targetRadius, RED);
            std::vector<Particle> hostParticles(numParticles);
            cudaMemcpy(hostParticles.data(), deviceParticles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

            for (const auto& particle : hostParticles) {
                if (particle.active) {
                    DrawCircle((int)particle.x, (int)particle.y, 2.0f,
                        { particle.r, particle.g, particle.b, particle.a });
                }
            }

            for (const auto& obstacle : obstacles) {
                DrawRectangle(obstacle.x, obstacle.y, obstacle.width, obstacle.height, GRAY);
            }

            DrawText("Move particles to the red target | Up/Down: change speed", 10, 10, 20, GRAY);
            DrawText(TextFormat("Score: %d", hostScore), 10, 70, 20, WHITE);
            DrawText(TextFormat("Speed: %.2f", speed), 10, 40, 20, GRAY);
            DrawText(timer.GetTimeLeft().c_str(), 10, 100, 20, WHITE);

            EndDrawing();

            // Timer expired
            if (timer.IsTimeUp()) break;
        }

        // Victory or defeat screen
        if (victory) {
            while (!WindowShouldClose() && !isTryAgain) {
                DrawVictoryScreen(screenWidth, screenHeight);
                if (IsKeyDown(KEY_R)) isTryAgain = true;
            }
        }
        else {
            while (!WindowShouldClose() && !isTryAgain) {
                DrawDefeatScreen(screenWidth, screenHeight);
                if (IsKeyDown(KEY_R)) isTryAgain = true;
            }
        }

        // Free resources
        cudaFree(deviceParticles);
        cudaFree(deviceObstacles);
        cudaFree(deviceScore);

    } while (isTryAgain);

    // Final cleanup
    UnloadMusicStream(music);
    CloseAudioDevice();
    CloseWindow();

    return 0;
}
