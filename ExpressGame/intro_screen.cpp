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


    Music music = LoadMusicStream("halo_theme.mp3");
    PlayMusicStream(music);

    char gametag[20] = ""; // Buffer for gametag input
    int currentDifficulty = 0; // 0: Easy, 1: Medium, 2: Hard

    while (!WindowShouldClose() && !result.startGame) {

        UpdateMusicStream(music);
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
        DrawTexture(logo, GetScreenWidth() / 2 - logo.width / 2, 250, WHITE);

        // Draw difficulty selection
        const char* difficulties[] = { "Easy", "Medium", "Hard" };
        Color difficultyColors[] = { GREEN, BLUE, RED };

        const int difficultyY = GetScreenHeight() / 2 + 120; // Adjusted position
        const int difficultySpacing = 220; // Spacing between difficulty texts
        const int startX = GetScreenWidth() / 2 - difficultySpacing; // Adjusted centering
        DrawText(
            difficulties[0],
            startX + 0* difficultySpacing,
            difficultyY-30,
            30,
            (0 == currentDifficulty) ? difficultyColors[0] : GRAY
        );
        DrawText(
            difficulties[1],
            startX + 1 * difficultySpacing-45,
            difficultyY - 30,
            30,
            (1 == currentDifficulty) ? difficultyColors[1] : GRAY
        );
        DrawText(
            difficulties[2],
            startX + 2 * difficultySpacing-60,
            difficultyY - 30,
            30,
            (2 == currentDifficulty) ? difficultyColors[2] : GRAY
        );

        DrawText(
            "Use Arrow Keys (Left/Right)",
            GetScreenWidth() / 2 - MeasureText("Use Arrow Keys (Left/Right)", 20) / 2,
            difficultyY+20,
            20,
            ORANGE
        );

        DrawText(
            "Press ENTER key when you are ready",
            GetScreenWidth() / 2 - MeasureText("Press ENTER key when you are ready", 30) / 2,
            difficultyY + 100,
            30,
            WHITE
        );

        
        
        

        EndDrawing();
    }
    UnloadMusicStream(music);
    return result;
}
