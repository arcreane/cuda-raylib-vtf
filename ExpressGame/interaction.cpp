#include "interaction.hpp"

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