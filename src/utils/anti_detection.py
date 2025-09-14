# =====================================
# utils/anti_detection.py
# =====================================
"""
Anti-detection techniques for web scraping
"""

import random
import time
from typing import List
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

class AntiDetection:
    """Collection of anti-detection techniques"""
    
    @staticmethod
    def add_mouse_entropy(driver: webdriver.Chrome):
        """Add realistic mouse movements"""
        
        actions = ActionChains(driver)
        body = driver.find_element(By.TAG_NAME, 'body')
        
        # Generate smooth curve points
        points = []
        for _ in range(5):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            points.append((x, y))
        
        # Move through points smoothly
        for x, y in points:
            actions.move_to_element_with_offset(body, x, y)
            actions.perform()
            time.sleep(random.uniform(0.1, 0.3))
    
    @staticmethod
    def type_like_human(element, text: str):
        """Type with human-like delays and patterns"""
        
        for i, char in enumerate(text):
            element.send_keys(char)
            
            # Variable delays
            if char == ' ':
                delay = random.uniform(0.1, 0.2)
            elif char in '.,!?':
                delay = random.uniform(0.2, 0.4)
            elif i > 0 and text[i-1] == char:  # Same key twice
                delay = random.uniform(0.15, 0.25)
            else:
                delay = random.uniform(0.05, 0.15)
            
            time.sleep(delay)
    
    @staticmethod
    def random_scroll(driver: webdriver.Chrome):
        """Perform random scrolling"""
        
        # Scroll down randomly
        scroll_height = random.randint(100, 500)
        driver.execute_script(f"window.scrollBy(0, {scroll_height});")
        time.sleep(random.uniform(0.5, 1.5))
        
        # Sometimes scroll back up
        if random.random() < 0.3:
            scroll_up = random.randint(50, 200)
            driver.execute_script(f"window.scrollBy(0, -{scroll_up});")
