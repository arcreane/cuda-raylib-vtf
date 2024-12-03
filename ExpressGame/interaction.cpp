#include "interaction.hpp"

void ProcessUserInput(float& speed, float& mouseX, float& mouseY, bool& attract, bool& repel) {
    // Contr�le de la vitesse
    if (IsKeyDown(KEY_UP)) speed += 0.1f;   // Augmenter la vitesse
    if (IsKeyDown(KEY_DOWN)) speed -= 0.1f; // R�duire la vitesse
    if (speed < 0.1f) speed = 0.1f;         // Vitesse minimale

    // G�rer la souris
    mouseX = GetMouseX();
    mouseY = GetMouseY();
    attract = IsMouseButtonDown(MOUSE_BUTTON_LEFT); // Attraction
    repel = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);  // R�pulsion
}