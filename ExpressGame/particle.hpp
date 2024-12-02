#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <vector>
#include <raylib.h> // For Color type
#include "utils.hpp"

// Particle structure to hold position, velocity, and color data
struct Particle {
    float x, y;       // Position
    float vx, vy;     // Velocity
    Color color;      // Color of the particle
};

// Function prototypes
void initialize_particles(std::vector<Particle>& particles, int num_particles);
void reset_particle(Particle& particle, float screen_width, float screen_height);

#endif // PARTICLE_HPP
