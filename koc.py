import cv2
import numpy as np

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

# Inicjalizacja detektora ruchu
detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# Inicjalizacja algorytmu śledzenia KLT
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Kolory punktów KLT (maksymalnie 100 różnych kolorów)
color = [(0,255,0) for _ in range(100)]

# Licznik osób
person_id = 0
person_locations = {}

# Inicjalizacja poprzedniej klatki
prev_frame = None

while True:
    # Wczytaj obraz z kamery
    ret, frame = cap.read()

    # Zastosuj detektor ruchu do obrazu
    mask = detector.apply(frame)

    # Usuń szumy i elementy niewielkich rozmiarów
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)

    # Znajdź kontury na obrazie
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicjalizacja listy punktów do śledzenia
    p0 = []

    # Zaznacz prostokąty wokół obszarów z ruchem i śledź je
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 4000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dodaj punkty śledzenia w środku prostokąta
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            p0.append([[cx, cy]])

            # Sprawdź czy osoba została już wcześniej zidentyfikowana
            assigned_person = None
            for person, (prev_x, prev_y) in person_locations.items():
                if abs(cx - prev_x) < 50 and abs(cy - prev_y) < 50:
                    assigned_person = person
                    break

            # Jeśli osoba została zidentyfikowana wcześniej, nadaj jej ten sam numer
            if assigned_person is not None:
                person_id = assigned_person
            else:
                person_locations[person_id] = (cx, cy)

            # Wypisz numer osoby na obrazie
            cv2.putText(frame, f'Person {person_id}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            person_id += 1

    # Zastosuj algorytm KLT do śledzenia punktów
    if len(p0) > 0 and prev_frame is not None:
        p0 = np.array(p0, dtype=np.float32)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
        for i, (new, old) in enumerate(zip(p1, p0)):
            a, b = new.ravel().astype(int)  # Zmień typ danych na integer
            c, d = old.ravel().astype(int)  # Zmień typ danych na integer
            mask = cv2.line(mask, (a, b), (c, d), color[i], 2)
            frame = cv2.circle(frame, (a, b), 5, color[i], -1)

    # Wyświetl obraz z zaznaczonymi prostokątami i numerami osób
    cv2.imshow('Motion Detection', frame)

    # Przechowaj poprzednią klatkę obrazu
    prev_frame = frame.copy()

    # Zakończ pętlę, jeśli naciśnięto klawisz 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij obiekty kamery i zamknij okna OpenCV
cap.release()
cv2.destroyAllWindows()