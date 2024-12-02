#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <raylib.h>

#include "particle.hpp"
#include "renderer.hpp"
#include "interaction.hpp"
#include "utils.hpp"

// Constants for circular motion
const float radius = 200.0f; // Radius of the circular path
const float center_x = 400.0f; // X position of the center
const float center_y = 300.0f; // Y position of the center
const float speed = 0.5f; // Speed of particles around the circle

// CUDA kernel to update particle positions for circular motion
__global__ void update_particles(Particle* particles, int num_particles, float delta_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_particles) {
        Particle& particle = particles[idx];

        // Calculate the angle of the particle with respect to the center
        float dx = particle.x - center_x;
        float dy = particle.y - center_y;
        float angle = atan2f(dy, dx);

        // Calculate new velocity components to move in a circular path
        float vx = -sin(angle) * speed;
        float vy = cos(angle) * speed;

        // Update particle velocity (keeping the speed constant)
        particle.vx = vx;
        particle.vy = vy;

        // Update particle position based on velocity
        particle.x += particle.vx * delta_time;
        particle.y += particle.vy * delta_time;
    }
}

// Helper function to check for CUDA errors
void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    const int num_particles = 1000;
    std::vector<Particle> particles(num_particles);

    // Initialize particles in random positions around the circle
    for (int i = 0; i < num_particles; ++i) {
        float angle = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159f; // Random angle
        particles[i].x = center_x + radius * cos(angle);
        particles[i].y = center_y + radius * sin(angle);
        particles[i].vx = -sin(angle) * speed; // Initial tangential velocity
        particles[i].vy = cos(angle) * speed; // Initial tangential velocity
    }

    // Initialize Raylib renderer
    Renderer renderer(800, 600, "CUDA Particle Circular Motion");

    // Allocate memory for CUDA on the device
    Particle* d_particles;
    check_cuda_error(cudaMalloc((void**)&d_particles, num_particles * sizeof(Particle)));

    // Copy initial data to device
    check_cuda_error(cudaMemcpy(d_particles, particles.data(), num_particles * sizeof(Particle), cudaMemcpyHostToDevice));

    bool simulation_running = true;
    const float delta_time = 1.0f / 60.0f; // Target 60 FPS

    while (!WindowShouldClose()) {
        handle_mouse_input(renderer);
        handle_keyboard_input(renderer, simulation_running);

        // Launch CUDA kernel to update particle positions for circular motion
        int threads_per_block = 256;
        int num_blocks = (num_particles + threads_per_block - 1) / threads_per_block;
        update_particles << <num_blocks, threads_per_block >> > (d_particles, num_particles, delta_time);
        check_cuda_error(cudaGetLastError());  // Check for kernel errors
        check_cuda_error(cudaDeviceSynchronize());  // Synchronize to ensure kernel has finished

        // Copy updated particles back to host
        check_cuda_error(cudaMemcpy(particles.data(), d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost));

        // Render particles
        renderer.draw_particles(particles);
        renderer.display();
    }

    // Clean up CUDA memory
    check_cuda_error(cudaFree(d_particles));

    // Close the Raylib window
    CloseWindow();

    return 0;
}
