import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Lightbulb, MessageSquare } from 'lucide-react'

export default function SuggestionCards() {
  // 'suggestions' array is passed as a prop from Python
  const suggestionList = props.suggestions || []
  
  const handleCardClick = (suggestion) => {
    // Use Chainlit's global API to send the suggestion as a user message
    sendUserMessage(suggestion)
  }
  
  return (
    <div className="flex flex-col gap-3 mt-2">
      {suggestionList.map((suggestion, index) => (
        <Card 
          key={index} 
          className="cursor-pointer hover:bg-accent/50 transition-colors border-border/40"
        >
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <div className="bg-primary/10 p-2 rounded-full">
                <Lightbulb className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1">
                <p className="text-sm mb-2">{suggestion}</p>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="h-7 px-2 text-xs"
                  onClick={() => handleCardClick(suggestion)}
                >
                  <MessageSquare className="h-3 w-3 mr-1" />
                  Ask this
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}