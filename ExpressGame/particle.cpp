#include "particle.hpp"
#include <cstdlib>  // For rand()
#include <ctime>    // For seeding random number generator

// Initialize particles with random positions, velocities, and colors
void initialize_particles(std::vector<Particle>& particles, int num_particles) {
    particles.resize(num_particles);
    srand(static_cast<unsigned>(time(nullptr))); // Seed random number generator

    for (auto& particle : particles) {
        reset_particle(particle, 800.0f, 600.0f); // Assuming default screen size
    }
}

// Reset a particle to a random position, velocity, and color
void reset_particle(Particle& particle, float screen_width, float screen_height) {
    particle.x = random_float(0.0f, screen_width);
    particle.y = random_float(0.0f, screen_height);

    particle.vx = 2; // Random velocity between -1.0 and 1.0
    particle.vy = -2;

    particle.color = { static_cast<unsigned char>(rand() % 256),
                       static_cast<unsigned char>(rand() % 256),
                       static_cast<unsigned char>(rand() % 256),
                       255 }; // Random color with full alpha
}

