package com.example.capstone_design.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import java.time.LocalDateTime;

@Getter @AllArgsConstructor
public class UserResponse {
    private Long id;
    private String username;
    private String role;
    private boolean enabled;
    private LocalDateTime createdAt;
}
