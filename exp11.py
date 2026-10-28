"""
Ex. No: 11 - HAND GESTURE RECOGNITION TO CONTROL AN ARCADE GAME
"""
import random
import sys
import cv2
import pygame

# 1. Initialize Pygame & Window
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Arcade Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# Colors
BLACK = (20, 20, 30)
WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
GREEN = (50, 230, 120)
RED = (255, 70, 70)

# Game Entities
paddle_width, paddle_height = 120, 18
paddle_x = WIDTH // 2 - paddle_width // 2
paddle_y = HEIGHT - 55

ball_radius = 15
ball_x = random.randint(30, WIDTH - 30)
ball_y = 60
ball_speed = 5
score = 0
game_over = False

# 2. Hand Tracking Setup (MediaPipe)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
except Exception:
    hands = None

cap = cv2.VideoCapture(0)
has_webcam = cap.isOpened()
if not has_webcam:
    print("Notice: Webcam not detected. Falling back to mouse tracking for paddle.")

# 3. Game Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Track paddle position via index finger or mouse fallback
    if has_webcam:
        ret, frame = cap.read()
        if ret and hands:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                fingertip = hand.landmark[8]
                paddle_x = int(fingertip.x * WIDTH) - paddle_width // 2
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False
    else:
        # Mouse fallback if no camera
        mouse_x, _ = pygame.mouse.get_pos()
        paddle_x = mouse_x - paddle_width // 2

    paddle_x = max(0, min(WIDTH - paddle_width, paddle_x))

    # Update Ball
    if not game_over:
        ball_y += ball_speed
        # Paddle Collision
        if (paddle_y <= ball_y + ball_radius <= paddle_y + paddle_height and 
            paddle_x <= ball_x <= paddle_x + paddle_width):
            score += 1
            ball_y = 25
            ball_x = random.randint(25, WIDTH - 25)
            ball_speed = min(ball_speed + 0.2, 12)
        # Missed Ball
        if ball_y - ball_radius > HEIGHT:
            game_over = True

    # Render Graphics
    screen.fill(BLACK)
    pygame.draw.rect(screen, BLUE, (paddle_x, paddle_y, paddle_width, paddle_height), border_radius=8)
    pygame.draw.circle(screen, GREEN, (ball_x, int(ball_y)), ball_radius)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (20, 20))

    if game_over:
        msg = font.render("GAME OVER - Close window to exit", True, RED)
        screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2))

    pygame.display.flip()
    clock.tick(30)

# Cleanup
if has_webcam:
    cap.release()
    cv2.destroyAllWindows()
pygame.quit()
sys.exit()

