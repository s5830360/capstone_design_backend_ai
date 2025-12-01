package com.example.capstone_design.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import java.time.LocalDateTime;
import com.fasterxml.jackson.annotation.JsonIgnore;

@Entity
@Table(name = "user_account", uniqueConstraints = {
        @UniqueConstraint(name = "uk_useraccount_username", columnNames = "username")
})
@Getter @Setter
@NoArgsConstructor @AllArgsConstructor @Builder

public class UserAccount {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable=false, length=40)
    private String username;

    @JsonIgnore
    @Column(nullable=false, length=100)
    private String passwordHash;     // BCrypt 해시 저장

    @Column(nullable=false, length=20)
    @Builder.Default
    private String role = "USER";    // 간단히 ROLE_USER 대용

    @Builder.Default
    private boolean enabled = true;

    @CreationTimestamp
    @Column(nullable=false, updatable=false)
    private LocalDateTime createdAt;
}
