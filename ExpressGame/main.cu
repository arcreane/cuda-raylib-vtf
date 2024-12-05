#include "raylib.h"
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include "renderer.hpp"
#include "interaction.hpp"
#include "timer.hpp"
#include "particle_simulation.cuh"
#include "cursor.hpp"
#include "intro_screen.hpp"
#include <iostream>

int main() {
    /*
     * Constants
     */
    const int screenWidth = 800;
    const int screenHeight = 600;
   
    const int numObstacles = 3;           // Number of obstacles
    const float influenceRadius = 150.0f; // Mouse influence radius
    const float targetRadius = 25.0f;     // Target radius

    float duration = 2.0f;
    int numParticles = 1001;        // Number of particles


    
    char* themeMusic = "cc_red_alert.mp3";
    char* winMusic = "gta_mission_passed.mp3";
    char* loseMusic = "mission_failed_mw3.mp3";


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

    
    

    SetTargetFPS(60);

    // Main game loop
    bool isTryAgain;

    do {
        // Create intro screen
        IntroScreen intro("A Game About Particules ...", "cursor_entry.png");

        // Show intro screen
        IntroScreenResult result = intro.Show();

        switch (result.difficulty) {
        case 0:
            duration = 100;
            loseMusic = "game_over.mp3";
            winMusic = "cantina.mp3";
            themeMusic = "hyper.mp3";
            break;
        case 1:
            duration = 80;
            loseMusic = "mission_lost.mp3";
            winMusic = "CourseClear.mp3";
            themeMusic = "terran.mp3";
            break;
        case 2:
            duration = 60;
            loseMusic = "mission_failed_mw3.mp3";
            winMusic = "gta_mission_passed.mp3";
            themeMusic = "cc_red_alert.mp3";
            break;

        }
            

        Music music = LoadMusicStream(themeMusic);
        isTryAgain = false;
        // Initialize game state
        Timer timer(duration);
        PlayMusicStream(music);

        Cursor cursor = Cursor();
        cursor.texture = LoadTexture("cursor.png");
        cursor.rect = { 0.0f, 0.0f, 30.0f, 40.0f };
        cursor.position = { 0.0f, 0.0f };
        HideCursor();

        // CUDA memory allocations
        Particle* deviceParticles = InitializeParticlesGPU(numParticles, screenWidth, screenHeight, obstacles, numObstacles);
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
        float speed = 3.0f;
        bool victory = false;

        // Main game loop
        while (!WindowShouldClose()) {
            float mouseX = 0.0f, mouseY = 0.0f;
            bool attract = false, repel = false;

            

            // Update logic
            timer.Update();
            UpdateMusicStream(music);
            ProcessUserInput(speed, mouseX, mouseY, attract, repel);

            cursor.position = Vector2{mouseX, mouseY};
            
            // CUDA kernel call for particle updates
            int blockSize = 256;
            int numBlocks = (numParticles + blockSize - 1) / blockSize;
            UpdateParticles(deviceParticles, numParticles, deviceObstacles, numObstacles, mouseX, mouseY,
                targetX, targetY, targetRadius, attract, influenceRadius, deviceScore, speed);

            // Copy score from GPU and check victory
            cudaMemcpy(&hostScore, deviceScore, sizeof(int), cudaMemcpyDeviceToHost);
            if (hostScore >= numParticles - 1) {
                victory = true;
                break;
            }

            // Rendering
            BeginDrawing();
            ClearBackground(BLACK);

            DrawTextureRec(cursor.texture, cursor.rect, cursor.position, WHITE);


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
        //StopMusicStream(music);
        UnloadMusicStream(music);
        // Victory or defeat screen
        // Victory or defeat screen
        if (victory) {
            music = LoadMusicStream(winMusic);
            PlayMusicStream(music);
            while (!WindowShouldClose() && !isTryAgain) {
                DrawVictoryScreen(screenWidth, screenHeight);
                if (IsKeyDown(KEY_R)) isTryAgain = true;

                UpdateMusicStream(music); // Update music stream to ensure proper playback
            }
            UnloadMusicStream(music);
            //StopMusicStream(music);
        }
        else {
            music = LoadMusicStream(loseMusic);
            PlayMusicStream(music);
            while (!WindowShouldClose() && !isTryAgain) {
                DrawDefeatScreen(screenWidth, screenHeight);
                if (IsKeyDown(KEY_R)) isTryAgain = true;

                UpdateMusicStream(music); // Update music stream to ensure proper playback
            }
            //StopMusicStream(music);
            UnloadMusicStream(music);
        }

        // Free resources
        cudaFree(deviceParticles);
        cudaFree(deviceObstacles);
        cudaFree(deviceScore);

    } while (isTryAgain);

    // Final cleanup
    
    CloseAudioDevice();
    CloseWindow();

    return 0;
}
