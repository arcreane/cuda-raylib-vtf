<<<<<<< HEAD
#ifndef INTERACTION_HPP
#define INTERACTION_HPP

#include <vector>
#include "particle.hpp"
#include "renderer.hpp"

// Struct for user-controlled fields (attractive/repulsive)
struct InteractionField {
    float x, y;        // Position of the field
    float strength;    // Positive for attraction, negative for repulsion
    float radius;      // Effective radius of influence
};

// Function prototypes for interaction handling
void handle_mouse_input(Renderer& renderer, std::vector<InteractionField>& fields);
void handle_keyboard_input(Renderer& renderer, std::vector<InteractionField>& fields, bool& simulation_running);

void apply_interaction_fields(std::vector<Particle>& particles, const std::vector<InteractionField>& fields, float delta_time);

#endif // INTERACTION_HPP
=======
// Fonction pour gérer les entrées utilisateur
#include "raylib.h"
void ProcessUserInput(float& speed, float& mouseX, float& mouseY, bool& attract, bool& repel);
>>>>>>> main
