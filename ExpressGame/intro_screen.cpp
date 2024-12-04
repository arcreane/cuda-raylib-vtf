#include "intro_screen.hpp"
#include <iostream>

IntroScreen::IntroScreen(const std::string& title, const std::string& logoPath)
    : gameTitle(title) {
    logo = LoadTexture(logoPath.c_str());
}

IntroScreen::~IntroScreen() {
    UnloadTexture(logo);
}

IntroScreenResult IntroScreen::Show() {
    IntroScreenResult result;
    result.startGame = false;
    result.difficulty = 0;

    char gametag[20] = ""; // Buffer for gametag input
    int currentDifficulty = 0; // 0: Easy, 1: Medium, 2: Hard

    while (!WindowShouldClose() && !result.startGame) {
        // Input handling
        if (IsKeyPressed(KEY_ENTER)) {
            result.gametag = std::string(gametag);
            result.difficulty = currentDifficulty;
            result.startGame = true;
        }

        if (IsKeyPressed(KEY_RIGHT)) currentDifficulty = (currentDifficulty + 1) % 3;
        if (IsKeyPressed(KEY_LEFT)) currentDifficulty = (currentDifficulty + 2) % 3;

        // Drawing
        BeginDrawing();
        ClearBackground(BLACK);

        // Draw game title
        DrawText(gameTitle.c_str(), GetScreenWidth() / 2 - MeasureText(gameTitle.c_str(), 50) / 2, 50, 50, WHITE);

        // Draw logo
        DrawTexture(logo, GetScreenWidth() / 2 - logo.width / 2, 150, WHITE);

        // Draw difficulty selection
        const char* difficulties[] = { "Easy", "Medium", "Hard" };
        Color difficultyColors[] = { GREEN, BLUE, ORANGE };

        const int difficultyY = GetScreenHeight() / 2 + 50; // Adjusted position
        const int difficultySpacing = 220; // Spacing between difficulty texts
        const int startX = GetScreenWidth() / 2 - difficultySpacing; // Adjusted centering
        DrawText(
            difficulties[0],
            startX + 0* difficultySpacing,
            difficultyY,
            30,
            (0 == currentDifficulty) ? difficultyColors[0] : GRAY
        );
        DrawText(
            difficulties[1],
            startX + 1 * difficultySpacing-45,
            difficultyY,
            30,
            (1 == currentDifficulty) ? difficultyColors[1] : GRAY
        );
        DrawText(
            difficulties[2],
            startX + 2 * difficultySpacing-60,
            difficultyY,
            30,
            (2 == currentDifficulty) ? difficultyColors[2] : GRAY
        );
        
        // Draw gametag input prompt
        int gametagY = GetScreenHeight() - 200; // Add more space above textfield
        DrawText("Enter your gametag:", 100, gametagY, 20, WHITE);
        DrawRectangle(100, gametagY + 30, 400, 40, DARKGRAY); // Adjusted text field position
        DrawText(gametag, 310, gametagY + 35, 20, WHITE);

        // Handle gametag input
        int key = GetCharPressed();
        if (key >= 32 && key <= 125 && strlen(gametag) < 19) {
            int len = strlen(gametag);
            gametag[len] = (char)key;
            gametag[len + 1] = '\0';
        }
        if (IsKeyPressed(KEY_BACKSPACE) && strlen(gametag) > 0) {
            gametag[strlen(gametag) - 1] = '\0';
        }
        
        

        EndDrawing();
    }

    return result;
}
