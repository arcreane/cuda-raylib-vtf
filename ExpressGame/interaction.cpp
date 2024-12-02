#include "interaction.hpp"
#include <raylib.h>

// Handle mouse input to add interaction fields
void handle_mouse_input(Renderer& renderer, std::vector<InteractionField>& fields) {
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_position = GetMousePosition();
        InteractionField field = {
            mouse_position.x,
            mouse_position.y,
            100.0f,   // Default strength (attractive)
            50.0f     // Default radius
        };
        fields.push_back(field);
    }

    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
        Vector2 mouse_position = GetMousePosition();
        InteractionField field = {
            mouse_position.x,
            mouse_position.y,
            -100.0f,  // Default strength (repulsive)
            50.0f     // Default radius
        };
        fields.push_back(field);
    }
}

// Handle keyboard input for global controls
void handle_keyboard_input(Renderer& renderer, std::vector<InteractionField>& fields, bool& simulation_running) {
    if (IsKeyPressed(KEY_R)) {
        fields.clear(); // Reset interaction fields
    }

    if (IsKeyPressed(KEY_SPACE)) {
        simulation_running = !simulation_running; // Pause/Resume simulation
    }
}

// Apply interaction fields to particles
void apply_interaction_fields(std::vector<Particle>& particles, const std::vector<InteractionField>& fields, float delta_time) {
    for (auto& particle : particles) {
        for (const auto& field : fields) {
            // Calculate distance to the interaction field
            float dx = field.x - particle.x;
            float dy = field.y - particle.y;
            float distance_squared = dx * dx + dy * dy;

            if (distance_squared < field.radius * field.radius) {
                float distance = sqrtf(distance_squared);

                // Normalize the direction vector
                float nx = dx / distance;
                float ny = dy / distance;

                // Apply force proportional to the field's strength and inverse square of the distance
                float force = field.strength / (distance + 1.0f); // Avoid division by zero
                particle.vx += nx * force * delta_time;
                particle.vy += ny * force * delta_time;
            }
        }
    }
}
