#ifndef INTRO_SCREEN_HPP
#define INTRO_SCREEN_HPP

#include "raylib.h"
#include <string>

struct IntroScreenResult {
    std::string gametag;
    int difficulty; // 0: Easy, 1: Medium, 2: Hard
    bool startGame;
};

class IntroScreen {
public:
    IntroScreen(const std::string& title, const std::string& logoPath);
    ~IntroScreen();
    IntroScreenResult Show();

private:
    std::string gameTitle;
    Texture2D logo;
};

#endif // INTRO_SCREEN_HPP
